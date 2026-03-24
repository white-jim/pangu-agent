"""
意图识别模块
使用大模型进行意图+动作联合识别
"""
import json
import logging
import re
import time
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """意图类型枚举"""
    QA = "qa"
    NAVIGATION = "navigation"
    VISION = "vision"
    REGISTRATION = "registration"
    SYSTEM = "system"
    UNKNOWN = "unknown"
    
    def __str__(self):
        return self.value
    
    @property
    def description(self) -> str:
        descriptions = {
            IntentType.QA: "通用问答",
            IntentType.NAVIGATION: "导航导览",
            IntentType.VISION: "视觉交互",
            IntentType.REGISTRATION: "信息注册",
            IntentType.SYSTEM: "系统控制",
            IntentType.UNKNOWN: "未知意图"
        }
        return descriptions.get(self, "未知")


class ActionType(Enum):
    """动作类型枚举"""
    QA_KNOWLEDGE = "qa_knowledge"
    QA_GENERAL = "qa_general"
    
    NAV_SHOW_MAP = "nav_show_map"
    NAV_GUIDE = "nav_guide"
    NAV_QUERY_LOCATION = "nav_query_location"
    
    VISION_FACE_DETECT = "vision_face_detect"
    VISION_FACE_RECOGNIZE = "vision_face_recognize"
    VISION_FACE_REGISTER = "vision_face_register"
    VISION_SCENE_ANALYZE = "vision_scene_analyze"
    
    REG_FACE = "reg_face"
    REG_INFO = "reg_info"
    
    SYS_STATUS = "sys_status"
    SYS_CLEAR = "sys_clear"
    SYS_HELP = "sys_help"
    
    UNKNOWN = "unknown"
    
    @property
    def description(self) -> str:
        descriptions = {
            ActionType.QA_KNOWLEDGE: "知识库问答",
            ActionType.QA_GENERAL: "通用问答",
            ActionType.NAV_SHOW_MAP: "展示地图",
            ActionType.NAV_GUIDE: "导航带路",
            ActionType.NAV_QUERY_LOCATION: "位置查询",
            ActionType.VISION_FACE_DETECT: "人脸检测",
            ActionType.VISION_FACE_RECOGNIZE: "人脸识别",
            ActionType.VISION_FACE_REGISTER: "人脸注册",
            ActionType.VISION_SCENE_ANALYZE: "场景分析",
            ActionType.REG_FACE: "人脸注册",
            ActionType.REG_INFO: "信息登记",
            ActionType.SYS_STATUS: "系统状态",
            ActionType.SYS_CLEAR: "清空对话",
            ActionType.SYS_HELP: "帮助信息",
            ActionType.UNKNOWN: "未知动作"
        }
        return descriptions.get(self, "未知")


INTENT_ACTION_MAPPING = {
    IntentType.QA: [ActionType.QA_KNOWLEDGE, ActionType.QA_GENERAL],
    IntentType.NAVIGATION: [ActionType.NAV_SHOW_MAP, ActionType.NAV_GUIDE, ActionType.NAV_QUERY_LOCATION],
    IntentType.VISION: [ActionType.VISION_FACE_DETECT, ActionType.VISION_FACE_RECOGNIZE, 
                        ActionType.VISION_FACE_REGISTER, ActionType.VISION_SCENE_ANALYZE],
    IntentType.REGISTRATION: [ActionType.REG_FACE, ActionType.REG_INFO],
    IntentType.SYSTEM: [ActionType.SYS_STATUS, ActionType.SYS_CLEAR, ActionType.SYS_HELP],
}


@dataclass
class Slot:
    """槽位信息"""
    name: str
    value: Optional[str] = None
    required: bool = False
    description: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": self.value,
            "required": self.required,
            "description": self.description
        }


@dataclass
class ToolchainStep:
    """工具链步骤"""
    step: int
    module: str
    action: str
    description: str
    
    def to_dict(self) -> Dict:
        return {
            "step": self.step,
            "module": self.module,
            "action": self.action,
            "description": self.description
        }


@dataclass
class IntentResult:
    """意图识别结果"""
    intent_type: IntentType
    action_type: ActionType
    confidence: float
    slots: Dict[str, Slot] = field(default_factory=dict)
    reasoning: str = ""
    raw_text: str = ""
    matched_keywords: List[str] = field(default_factory=list)
    recognition_method: str = "unknown"
    llm_response: str = ""
    duration_ms: float = 0.0
    needs_toolchain: bool = False
    toolchain_plan: List[ToolchainStep] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "intent_type": self.intent_type.value,
            "intent_description": self.intent_type.description,
            "action_type": self.action_type.value,
            "action_description": self.action_type.description,
            "confidence": self.confidence,
            "slots": {k: v.to_dict() for k, v in self.slots.items()},
            "reasoning": self.reasoning,
            "raw_text": self.raw_text,
            "matched_keywords": self.matched_keywords,
            "recognition_method": self.recognition_method,
            "llm_response": self.llm_response,
            "duration_ms": self.duration_ms,
            "needs_toolchain": self.needs_toolchain,
            "toolchain_plan": [s.to_dict() for s in self.toolchain_plan]
        }
    
    def get_slot_value(self, slot_name: str) -> Optional[str]:
        """获取槽位值"""
        slot = self.slots.get(slot_name)
        return slot.value if slot else None


class IntentRecognizer:
    """
    意图识别器
    使用大模型进行意图+动作联合识别
    """
    
    INTENT_PROMPT = """你是一个智能助手意图识别专家。请仔细分析用户的输入，识别用户的意图、具体动作，并判断是否需要多步骤的工具链调用。

## 可用模块和动作：
1. navigation模块（导航导览）：
   - nav_guide: 导航带路到目标位置
   - nav_show_map: 展示地图
   - nav_query_location: 查询位置信息

2. vision模块（视觉交互）：
   - vision_face_detect: 检测人脸
   - vision_face_recognize: 识别人脸
   - vision_face_register: 注册人脸
   - vision_scene_analyze: 分析场景

3. semantic_interaction模块（语义问答）：
   - qa_knowledge: 知识库问答
   - qa_general: 通用问答

## 工具链分析（关键）：
仔细判断用户请求是否需要多个步骤按顺序执行。以下情况需要工具链：

【多步骤任务示例】
1. "去门口接待一下客人，把他带到接待室去，记得给他注册人脸"
   → 需要3步：导航到门口 → 导航到接待室 → 注册人脸

2. "去查看一下一号安全出口是否畅通"
   → 需要2步：导航到安全出口 → 场景分析

3. "带我去会议室，然后看看有谁在"
   → 需要2步：导航到会议室 → 人脸检测/识别

4. "注册张三的人脸，然后验证一下"
   → 需要2步：注册人脸 → 人脸识别验证

5. "去接待室看看有没有客人"
   → 需要2步：导航到接待室 → 人脸检测

【单步骤任务】
- "带我去会议室" → 只需导航
- "注册张三的人脸" → 只需注册
- "看看这是谁" → 只需识别

## 槽位提取：
- target_location: 目标位置（导航相关）
- person_name: 人名（人脸相关）
- location_name: 位置名称

## 输出格式（必须严格按JSON格式）：
{{
    "intent": "意图类型(navigation/vision/registration/qa/system)",
    "action": "主要动作类型",
    "confidence": 0.95,
    "reasoning": "详细的判断理由，说明为什么需要这些步骤",
    "slots": {{
        "target_location": "提取的位置",
        "person_name": "提取的人名"
    }},
    "needs_toolchain": true或false,
    "toolchain_plan": [
        {{
            "step": 1,
            "module": "模块名",
            "action": "动作名",
            "description": "具体执行内容"
        }},
        {{
            "step": 2,
            "module": "模块名",
            "action": "动作名",
            "description": "具体执行内容"
        }}
    ]
}}

用户输入：{query}

请直接输出JSON，不要有其他内容："""

    FALLBACK_KEYWORDS: Dict[IntentType, Dict[str, List[str]]] = {
        IntentType.NAVIGATION: {
            "guide": ["带我去", "带路", "导航到", "去", "前往", "走到", "怎么走", "领我去"],
            "show_map": ["地图", "展示地图", "看地图", "哪里有"],
            "query": ["在哪里", "位置在哪", "什么位置"]
        },
        IntentType.VISION: {
            "detect": ["检测人脸", "有人吗", "看看有没有人", "人脸检测"],
            "recognize": ["是谁", "认出", "识别这个人", "这个人是谁", "看看这是谁"],
            "register": ["注册人脸", "录入人脸", "添加人脸", "登记人脸"],
            "analyze": ["看看", "分析", "描述", "眼前", "场景", "这是什么"]
        },
        IntentType.REGISTRATION: {
            "face": ["注册人脸", "录入人脸", "添加人脸"],
            "info": ["登记", "注册信息", "录入信息"]
        },
        IntentType.SYSTEM: {
            "status": ["状态", "运行状态", "系统状态"],
            "clear": ["清空", "清除", "重置", "重新开始"],
            "help": ["帮助", "怎么用", "使用方法", "功能"]
        },
        IntentType.QA: {
            "knowledge": ["项目", "系统", "功能", "模块", "架构", "技术"],
            "general": []
        }
    }
    
    def __init__(self, model_client=None, threshold: float = 0.6):
        self.model_client = model_client
        self.threshold = threshold
        self._last_recognition_details: Dict = {}
    
    def set_model_client(self, client):
        """设置模型客户端"""
        self.model_client = client
    
    def get_last_details(self) -> Dict:
        """获取最近一次识别的详细信息"""
        return self._last_recognition_details
    
    def recognize(self, text: str) -> IntentResult:
        """
        识别用户意图
        
        Args:
            text: 用户输入文本
            
        Returns:
            IntentResult: 意图识别结果
        """
        text = text.strip()
        start_time = time.time()
        
        logger.info(f"="*60)
        logger.info(f"开始意图识别，输入: {text}")
        logger.info(f"模型客户端是否可用: {self.model_client is not None}")
        if self.model_client:
            logger.info(f"模型客户端类型: {type(self.model_client).__name__}")
        
        self._last_recognition_details = {
            "input_text": text,
            "method_attempted": "llm" if self.model_client else "rules",
            "llm_available": self.model_client is not None
        }
        
        if self.model_client:
            logger.info("使用大模型进行意图识别...")
            try:
                result = self._recognize_with_llm(text)
                duration = (time.time() - start_time) * 1000
                result.duration_ms = duration
                
                self._last_recognition_details.update({
                    "method_used": "llm",
                    "llm_success": True,
                    "llm_response": result.llm_response[:500] if result.llm_response else None,
                    "duration_ms": duration
                })
                
                logger.info(f"大模型识别结果: intent={result.intent_type.value}, action={result.action_type.value}, confidence={result.confidence:.2f}")
                logger.info(f"大模型工具链分析: needs_toolchain={result.needs_toolchain}, plan_count={len(result.toolchain_plan)}")
                logger.info(f"大模型意图识别成功，返回结果")
                return result
                    
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                self._last_recognition_details.update({
                    "method_used": "rules",
                    "llm_success": False,
                    "llm_error": str(e),
                    "duration_ms": duration
                })
                logger.error(f"大模型意图识别失败: {e}", exc_info=True)
                # 大模型失败时抛出异常，不使用规则匹配
                raise RuntimeError(f"大模型意图识别失败: {e}")
        else:
            error_msg = "模型客户端不可用，无法进行意图识别"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _recognize_with_llm(self, text: str) -> IntentResult:
        """使用大模型进行意图识别"""
        logger.info(f"正在调用大模型生成意图识别结果...")
        prompt = self.INTENT_PROMPT.format(query=text)
        
        try:
            logger.info(f"发送提示词到大模型...")
            response = self.model_client.generate(prompt)
            logger.info(f"大模型返回结果: {response[:200]}...")
            
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                logger.info(f"解析到JSON数据: {data}")
                
                intent_str = data.get("intent", "unknown")
                action_str = data.get("action", "unknown")
                confidence = float(data.get("confidence", 0.5))
                reasoning = data.get("reasoning", "")
                slots_data = data.get("slots", {})
                needs_toolchain = data.get("needs_toolchain", False)
                toolchain_data = data.get("toolchain_plan", [])
                
                try:
                    intent_type = IntentType(intent_str)
                except ValueError:
                    intent_type = IntentType.UNKNOWN
                    
                try:
                    action_type = ActionType(action_str)
                except ValueError:
                    action_type = ActionType.UNKNOWN
                
                slots = {}
                for name, value in slots_data.items():
                    if value:
                        slots[name] = Slot(name=name, value=str(value), required=True)
                
                toolchain_plan = []
                if needs_toolchain and toolchain_data:
                    for step_data in toolchain_data:
                        step = ToolchainStep(
                            step=step_data.get("step", 1),
                            module=step_data.get("module", ""),
                            action=step_data.get("action", ""),
                            description=step_data.get("description", "")
                        )
                        toolchain_plan.append(step)
                
                # 完全依赖大模型的识别结果，不使用规则推断补充
                if needs_toolchain and not toolchain_plan:
                    logger.warning(f"大模型识别需要工具链但未提供详细计划，将使用大模型识别的意图执行")
                
                logger.info(f"最终工具链分析: needs_toolchain={needs_toolchain}, plan_count={len(toolchain_plan)}")
                
                return IntentResult(
                    intent_type=intent_type,
                    action_type=action_type,
                    confidence=confidence,
                    slots=slots,
                    reasoning=reasoning,
                    raw_text=text,
                    recognition_method="llm",
                    llm_response=response,
                    needs_toolchain=needs_toolchain,
                    toolchain_plan=toolchain_plan
                )
            else:
                logger.warning(f"未在响应中找到JSON: {response}")
                raise ValueError(f"大模型返回格式错误，未找到JSON: {response[:200]}")
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}", exc_info=True)
            raise ValueError(f"JSON解析失败: {e}")
        except Exception as e:
            logger.error(f"大模型识别异常: {e}", exc_info=True)
            raise
    
    def _infer_toolchain_from_query(self, text: str, intent_type: IntentType, action_type: ActionType, slots: Dict) -> Tuple[bool, List[ToolchainStep]]:
        """根据查询内容推断工具链（规则匹配回退方案）"""
        text_lower = text.lower()
        needs_toolchain = False
        toolchain_plan = []
        
        # 获取槽位值
        target_location = ""
        person_name = ""
        for name, slot in slots.items():
            if name in ["target_location", "location_name"]:
                target_location = slot.value
            if name == "person_name":
                person_name = slot.value
        
        # 模式1: 接待客人（导航+导航+注册）
        if any(kw in text_lower for kw in ["接待", "客人", "带到", "带他去"]):
            needs_toolchain = True
            # 提取第一个位置（门口/入口）
            first_location = "门口"
            if "门口" in text_lower:
                first_location = "门口"
            elif "入口" in text_lower:
                first_location = "入口"
            
            # 提取第二个位置（接待室/会议室）
            second_location = target_location if target_location else "接待室"
            
            toolchain_plan = [
                ToolchainStep(
                    step=1,
                    module="navigation",
                    action="nav_guide",
                    description=f"导航到{first_location}接待客人"
                ),
                ToolchainStep(
                    step=2,
                    module="navigation",
                    action="nav_guide",
                    description=f"带客人到{second_location}"
                )
            ]
            # 如果有注册人脸的要求
            if any(kw in text_lower for kw in ["注册", "录入", "登记"]):
                toolchain_plan.append(
                    ToolchainStep(
                        step=3,
                        module="vision",
                        action="vision_face_register",
                        description=f"为客人注册人脸"
                    )
                )
            return needs_toolchain, toolchain_plan
        
        # 模式2: 去某处查看/检查（导航+视觉）
        if any(kw in text_lower for kw in ["去看看", "去查看", "去确认", "检查", "看看有没有"]):
            needs_toolchain = True
            location = target_location if target_location else "目标位置"
            
            # 判断是查看人还是查看场景
            if any(kw in text_lower for kw in ["谁", "人", "客人", "有人"]):
                vision_action = "vision_face_detect"
                vision_desc = "检测是否有人"
            else:
                vision_action = "vision_scene_analyze"
                vision_desc = "分析场景状态"
            
            toolchain_plan = [
                ToolchainStep(
                    step=1,
                    module="navigation",
                    action="nav_guide",
                    description=f"导航到{location}"
                ),
                ToolchainStep(
                    step=2,
                    module="vision",
                    action=vision_action,
                    description=vision_desc
                )
            ]
            return needs_toolchain, toolchain_plan
        
        # 模式3: 导航+查看（顺序执行）
        if ("导航" in text_lower or "带我去" in text_lower or "去" in text_lower) and \
           any(kw in text_lower for kw in ["看看", "查看", "然后看", "再看"]):
            needs_toolchain = True
            location = target_location if target_location else "目标位置"
            
            toolchain_plan = [
                ToolchainStep(
                    step=1,
                    module="navigation",
                    action="nav_guide",
                    description=f"导航到{location}"
                ),
                ToolchainStep(
                    step=2,
                    module="vision",
                    action="vision_scene_analyze",
                    description="分析当前场景"
                )
            ]
            return needs_toolchain, toolchain_plan
        
        # 模式4: 注册+验证
        if any(kw in text_lower for kw in ["注册.*然后验证", "录入.*然后识别", "添加.*然后确认", "注册.*再验证"]):
            needs_toolchain = True
            name = person_name if person_name else "用户"
            
            toolchain_plan = [
                ToolchainStep(
                    step=1,
                    module="vision",
                    action="vision_face_register",
                    description=f"注册{name}的人脸"
                ),
                ToolchainStep(
                    step=2,
                    module="vision",
                    action="vision_face_recognize",
                    description=f"验证{name}的身份"
                )
            ]
            return needs_toolchain, toolchain_plan
        
        return needs_toolchain, toolchain_plan
    
    def _recognize_with_rules(self, text: str) -> IntentResult:
        """基于规则的意图识别（备选方案）"""
        text_lower = text.lower()
        
        best_intent = IntentType.QA
        best_action = ActionType.QA_KNOWLEDGE
        best_confidence = 0.0
        best_keywords = []
        best_slots = {}
        
        for intent_type, actions in self.FALLBACK_KEYWORDS.items():
            for action_suffix, keywords in actions.items():
                matched = [kw for kw in keywords if kw in text_lower]
                
                if matched:
                    confidence = min(len(matched) * 0.3 + 0.4, 0.95)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = intent_type
                        best_keywords = matched
                        
                        action_name = f"{intent_type.value}_{action_suffix}" if "_" not in action_suffix else action_suffix
                        try:
                            best_action = ActionType(action_name)
                        except ValueError:
                            if intent_type == IntentType.NAVIGATION:
                                best_action = ActionType.NAV_GUIDE
                            elif intent_type == IntentType.VISION:
                                best_action = ActionType.VISION_SCENE_ANALYZE
                            else:
                                best_action = ActionType.UNKNOWN
                        
                        best_slots = self._extract_slots(text, best_intent, best_action)
        
        if best_confidence < self.threshold:
            best_intent = IntentType.QA
            best_action = ActionType.QA_KNOWLEDGE
            best_confidence = 0.5
        
        needs_toolchain, toolchain_plan = self._infer_toolchain_from_query(text, best_intent, best_action, best_slots)
        
        logger.info(f"规则匹配工具链分析: needs_toolchain={needs_toolchain}, plan_count={len(toolchain_plan)}")
        
        return IntentResult(
            intent_type=best_intent,
            action_type=best_action,
            confidence=best_confidence,
            slots=best_slots,
            reasoning=f"基于关键词匹配: {', '.join(best_keywords) if best_keywords else '默认问答'}",
            raw_text=text,
            matched_keywords=best_keywords,
            needs_toolchain=needs_toolchain,
            toolchain_plan=toolchain_plan
        )
    
    def _extract_slots(self, text: str, intent: IntentType, action: ActionType) -> Dict[str, Slot]:
        """提取槽位信息"""
        slots = {}
        
        if intent == IntentType.NAVIGATION:
            patterns = [
                (r"去(.+?)(?:怎么走|吧|。|$)", "target_location"),
                (r"导航到(.+?)(?:吧|。|$)", "target_location"),
                (r"带我去(.+?)(?:吧|。|$)", "target_location"),
                (r"(.+?)在哪里", "target_location"),
            ]
            for pattern, slot_name in patterns:
                match = re.search(pattern, text)
                if match:
                    slots[slot_name] = Slot(
                        name=slot_name,
                        value=match.group(1).strip(),
                        required=True,
                        description="目标位置"
                    )
                    break
        
        elif intent == IntentType.VISION:
            patterns = [
                (r"识别(.+?)的脸", "person_name"),
                (r"(.+?)是谁", "person_name"),
                (r"注册(.+?)的人脸", "person_name"),
            ]
            for pattern, slot_name in patterns:
                match = re.search(pattern, text)
                if match:
                    slots[slot_name] = Slot(
                        name=slot_name,
                        value=match.group(1).strip(),
                        required=True,
                        description="人名"
                    )
                    break
        
        return slots
    
    def get_supported_intents(self) -> List[Dict]:
        """获取支持的意图类型"""
        return [
            {
                "type": intent.value,
                "description": intent.description,
                "actions": [action.value for action in INTENT_ACTION_MAPPING.get(intent, [])]
            }
            for intent in IntentType
        ]
