"""
智能决策中枢
核心调度逻辑，支持意图+动作联合决策
"""
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from .intent_recognizer import IntentRecognizer, IntentType, ActionType, IntentResult
from .module_registry import ModuleRegistry, BaseModule, ModuleResult
from .context_manager import ContextManager, DialogueContext

logger = logging.getLogger(__name__)


@dataclass
class DecisionStep:
    """决策步骤"""
    step_name: str
    step_description: str
    status: str = "pending"
    result: Any = None
    success: bool = True
    duration_ms: float = 0.0
    details: Dict = field(default_factory=dict)
    sub_steps: List['DecisionStep'] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "step_name": self.step_name,
            "step_description": self.step_description,
            "status": self.status,
            "result": self.result,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "details": self.details,
            "sub_steps": [s.to_dict() for s in self.sub_steps]
        }


@dataclass
class ToolchainExecutionResult:
    """工具链执行结果"""
    step_number: int
    module_name: str
    action: str
    success: bool
    result: Any
    message: str
    duration_ms: float
    
    def to_dict(self) -> Dict:
        return {
            "step_number": self.step_number,
            "module_name": self.module_name,
            "action": self.action,
            "success": self.success,
            "result": self.result,
            "message": self.message,
            "duration_ms": self.duration_ms
        }


@dataclass
class DecisionResult:
    """决策结果"""
    success: bool
    response: str
    intent: IntentResult
    module_name: str
    action_taken: str
    steps: List[DecisionStep] = field(default_factory=list)
    context: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    toolchain_results: List[ToolchainExecutionResult] = field(default_factory=list)
    is_toolchain_execution: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "response": self.response,
            "intent": self.intent.to_dict(),
            "module_name": self.module_name,
            "action_taken": self.action_taken,
            "steps": [s.to_dict() for s in self.steps],
            "context": self.context,
            "metadata": self.metadata,
            "toolchain_results": [r.to_dict() for r in self.toolchain_results],
            "is_toolchain_execution": self.is_toolchain_execution
        }


class DecisionCenter:
    """
    智能决策中枢
    负责意图识别、动作决策、模块调度、上下文管理和结果整合
    """
    
    ACTION_MODULE_MAPPING = {
        ActionType.QA_KNOWLEDGE: "semantic_interaction",
        ActionType.QA_GENERAL: "semantic_interaction",
        ActionType.NAV_SHOW_MAP: "navigation",
        ActionType.NAV_GUIDE: "navigation",
        ActionType.NAV_QUERY_LOCATION: "navigation",
        ActionType.VISION_FACE_DETECT: "vision",
        ActionType.VISION_FACE_RECOGNIZE: "vision",
        ActionType.VISION_FACE_REGISTER: "vision",
        ActionType.VISION_SCENE_ANALYZE: "vision",
        ActionType.REG_FACE: "vision",
        ActionType.REG_INFO: "semantic_interaction",
        ActionType.SYS_STATUS: "semantic_interaction",
        ActionType.SYS_CLEAR: "semantic_interaction",
        ActionType.SYS_HELP: "semantic_interaction",
        ActionType.UNKNOWN: "semantic_interaction",
    }
    
    def __init__(
        self,
        intent_threshold: float = 0.6,
        max_context_turns: int = 5,
        model_client: Optional[Any] = None
    ):
        self.intent_threshold = intent_threshold
        self.max_context_turns = max_context_turns
        self.model_client = model_client
        
        self.intent_recognizer = IntentRecognizer(
            model_client=model_client,
            threshold=intent_threshold
        )
        self.module_registry = ModuleRegistry.get_instance()
        self.context_manager = ContextManager(max_context_turns=max_context_turns)
        
        self._preprocessors: List[Callable] = []
        self._postprocessors: List[Callable] = []
        
        self._initialized = False
    
    def set_model_client(self, client: Any):
        """设置模型客户端"""
        self.model_client = client
        self.intent_recognizer.set_model_client(client)
    
    def initialize(self):
        """初始化决策中枢"""
        if self._initialized:
            return
        
        logger.info("初始化智能决策中枢...")
        
        init_results = self.module_registry.initialize_all()
        for name, success in init_results.items():
            if not success:
                logger.warning(f"模块 {name} 初始化失败")
        
        self._initialized = True
        logger.info("智能决策中枢初始化完成")
    
    def register_module(self, module: BaseModule) -> bool:
        """注册功能模块"""
        return self.module_registry.register(module)
    
    def add_preprocessor(self, processor: Callable):
        """添加预处理器"""
        self._preprocessors.append(processor)
    
    def add_postprocessor(self, processor: Callable):
        """添加后处理器"""
        self._postprocessors.append(processor)
    
    def process(self, user_input: str, session_id: str = "default") -> DecisionResult:
        """处理用户输入"""
        steps: List[DecisionStep] = []
        
        for processor in self._preprocessors:
            try:
                user_input = processor(user_input)
            except Exception as e:
                logger.warning(f"预处理器执行失败: {e}")
        
        step = DecisionStep(
            step_name="input_preprocessing",
            step_description="预处理用户输入",
            status="completed",
            result={"original_input": user_input, "processed": True},
            success=True
        )
        steps.append(step)
        
        start_time = time.time()
        intent_step = DecisionStep(
            step_name="intent_recognition",
            step_description="正在使用大模型解析用户意图和动作...",
            status="running",
            sub_steps=[]
        )
        steps.append(intent_step)
        
        intent_sub_step = DecisionStep(
            step_name="llm_intent_call",
            step_description=f"调用大模型进行意图识别（{'已配置' if self.model_client else '未配置，使用规则匹配'}）",
            status="running"
        )
        intent_step.sub_steps.append(intent_sub_step)
        
        intent_result = self.intent_recognizer.recognize(user_input)
        duration = (time.time() - start_time) * 1000
        
        intent_sub_step.status = "completed"
        intent_sub_step.success = True
        intent_sub_step.duration_ms = duration
        intent_sub_step.result = {
            "method": intent_result.recognition_method,
            "llm_response": intent_result.llm_response[:500] if intent_result.llm_response else None
        }
        
        parse_sub_step = DecisionStep(
            step_name="intent_parsing",
            step_description="解析大模型输出，提取意图和动作",
            status="completed",
            success=True,
            result={
                "intent_type": intent_result.intent_type.value,
                "action_type": intent_result.action_type.value,
                "confidence": intent_result.confidence,
                "reasoning": intent_result.reasoning
            }
        )
        intent_step.sub_steps.append(parse_sub_step)
        
        slots_sub_step = DecisionStep(
            step_name="slot_extraction",
            step_description="提取槽位信息",
            status="completed",
            success=True,
            result={k: v.to_dict() for k, v in intent_result.slots.items()} if intent_result.slots else {}
        )
        intent_step.sub_steps.append(slots_sub_step)
        
        steps[-1].status = "completed"
        steps[-1].success = True
        steps[-1].duration_ms = duration
        steps[-1].result = intent_result.to_dict()
        steps[-1].details = self.intent_recognizer.get_last_details()
        steps[-1].step_description = f"意图识别完成：{intent_result.intent_type.description} / {intent_result.action_type.description}（置信度: {intent_result.confidence:.2f}）"
        
        action_step = DecisionStep(
            step_name="action_decision",
            step_description="动作决策：确定执行模块和具体动作",
            status="running",
            sub_steps=[]
        )
        steps.append(action_step)
        
        module_name = self._get_module_by_action(intent_result.action_type)
        
        module_lookup_step = DecisionStep(
            step_name="module_lookup",
            step_description=f"根据动作类型 {intent_result.action_type.value} 查找对应模块",
            status="completed",
            success=True,
            result={
                "action_type": intent_result.action_type.value,
                "mapped_module": module_name
            }
        )
        action_step.sub_steps.append(module_lookup_step)
        
        action_params_step = DecisionStep(
            step_name="action_params",
            step_description="准备模块执行参数",
            status="completed",
            success=True,
            result={
                "query": user_input,
                "intent": intent_result.intent_type.value,
                "action": intent_result.action_type.value,
                "slots": {k: v.value for k, v in intent_result.slots.items()}
            }
        )
        action_step.sub_steps.append(action_params_step)
        
        action_step.status = "completed"
        action_step.success = True
        action_step.result = {
            "module": module_name,
            "action": intent_result.action_type.value,
            "slots": {k: v.to_dict() for k, v in intent_result.slots.items()}
        }
        action_step.step_description = f"动作决策完成：调用 {module_name} 模块执行 {intent_result.action_type.description}"
        
        toolchain_results: List[ToolchainExecutionResult] = []
        is_toolchain_execution = False
        
        if intent_result.needs_toolchain and intent_result.toolchain_plan:
            is_toolchain_execution = True
            toolchain_step = DecisionStep(
                step_name="toolchain_analysis",
                step_description=f"工具链分析：检测到需要 {len(intent_result.toolchain_plan)} 个步骤",
                status="running",
                sub_steps=[]
            )
            steps.append(toolchain_step)
            
            for i, tc_step in enumerate(intent_result.toolchain_plan, 1):
                tc_sub_step = DecisionStep(
                    step_name=f"toolchain_step_{i}",
                    step_description=f"步骤{i}: {tc_step.module} 模块 - {tc_step.description}",
                    status="completed",
                    success=True,
                    result={
                        "step": tc_step.step,
                        "module": tc_step.module,
                        "action": tc_step.action,
                        "description": tc_step.description
                    }
                )
                toolchain_step.sub_steps.append(tc_sub_step)
            
            toolchain_step.status = "completed"
            toolchain_step.success = True
            toolchain_step.step_description = f"工具链规划完成：{len(intent_result.toolchain_plan)} 个执行步骤"
        
        context = self.context_manager.get_context_for_model(session_id)
        
        step = DecisionStep(
            step_name="context_retrieval",
            step_description=f"检索对话上下文：找到 {len(context)} 条历史记录",
            status="completed",
            result={"context_length": len(context)},
            success=True
        )
        steps.append(step)
        
        # 判断是否执行工具链
        if is_toolchain_execution and intent_result.toolchain_plan:
            # 执行工具链
            response, toolchain_results = self._execute_toolchain(
                user_input=user_input,
                toolchain_plan=intent_result.toolchain_plan,
                context=context,
                session_id=session_id,
                intent_result=intent_result,
                steps=steps
            )
        else:
            # 单模块执行
            module = self.module_registry.get_module(module_name)
            
            if module is None:
                module = self.module_registry.get_module("semantic_interaction")
                if module is None:
                    step = DecisionStep(
                        step_name="module_execution",
                        step_description="模块执行",
                        status="failed",
                        result={"error": "未找到可用模块"},
                        success=False
                    )
                    steps.append(step)
                    return DecisionResult(
                        success=False,
                        response="抱歉，系统暂时无法处理您的请求。",
                        intent=intent_result,
                        module_name=module_name,
                        action_taken="none",
                        steps=steps,
                        toolchain_results=[],
                        is_toolchain_execution=False
                    )
            
            start_time = time.time()
            step = DecisionStep(
                step_name="module_execution",
                step_description=f"正在执行 {module_name} 模块...",
                status="running"
            )
            steps.append(step)
            
            module_result = None
            try:
                module_result = module.execute(
                    query=user_input,
                    context={
                        "history": context,
                        "session_id": session_id,
                        "intent": intent_result.intent_type.value,
                        "action": intent_result.action_type.value,
                        "slots": {k: v.value for k, v in intent_result.slots.items()}
                    }
                )
                duration = (time.time() - start_time) * 1000
                
                steps[-1].status = "completed"
                steps[-1].success = module_result.success
                steps[-1].duration_ms = duration
                steps[-1].result = module_result.to_dict()
                steps[-1].details = module_result.metadata
                steps[-1].step_description = f"{module_name} 模块执行完成"
                
                response = module_result.data if module_result.success else module_result.message
                
            except Exception as e:
                logger.error(f"模块执行失败: {e}")
                duration = (time.time() - start_time) * 1000
                steps[-1].status = "failed"
                steps[-1].success = False
                steps[-1].duration_ms = duration
                steps[-1].result = {"error": str(e)}
                steps[-1].step_description = f"{module_name} 模块执行失败"
                
                response = f"处理您的请求时发生错误: {str(e)}"
                module_result = ModuleResult(success=False, data=None, message=response)
        
        self.context_manager.add_turn(
            session_id=session_id,
            user_input=user_input,
            intent=intent_result.intent_type.value,
            module_name=module_name,
            response=response
        )
        
        step = DecisionStep(
            step_name="context_update",
            step_description="更新对话上下文",
            status="completed",
            result={"session_id": session_id},
            success=True
        )
        steps.append(step)
        
        for processor in self._postprocessors:
            try:
                response = processor(response)
            except Exception as e:
                logger.warning(f"后处理器执行失败: {e}")
        
        # 确定最终执行状态和元数据
        if is_toolchain_execution:
            # 工具链执行：根据所有步骤结果确定最终状态
            all_success = all(r.success for r in toolchain_results)
            final_metadata = {
                "needs_toolchain": True,
                "toolchain_plan": [s.to_dict() for s in intent_result.toolchain_plan],
                "toolchain_step_count": len(intent_result.toolchain_plan),
                "toolchain_completed_count": len([r for r in toolchain_results if r.success]),
                "toolchain_all_success": all_success
            }
            final_success = all_success
            final_module_name = "toolchain"
        else:
            # 单模块执行
            final_metadata = module_result.metadata.copy() if module_result else {}
            final_success = module_result.success if module_result else False
            final_module_name = module_name
        
        return DecisionResult(
            success=final_success,
            response=response,
            intent=intent_result,
            module_name=final_module_name,
            action_taken=intent_result.action_type.value,
            steps=steps,
            context={"session_id": session_id},
            metadata=final_metadata,
            toolchain_results=toolchain_results,
            is_toolchain_execution=is_toolchain_execution
        )
    
    def _get_module_by_action(self, action_type: ActionType) -> str:
        """根据动作类型获取模块名称"""
        module_name = self.ACTION_MODULE_MAPPING.get(action_type, "semantic_interaction")
        
        module = self.module_registry.get_module(module_name)
        if module:
            return module_name
        
        return "semantic_interaction"
    
    def _execute_toolchain(
        self,
        user_input: str,
        toolchain_plan: List[Any],
        context: List[Dict],
        session_id: str,
        intent_result: IntentResult,
        steps: List[DecisionStep]
    ) -> tuple:
        """
        执行工具链
        
        Args:
            user_input: 用户输入
            toolchain_plan: 工具链计划
            context: 对话上下文
            session_id: 会话ID
            intent_result: 意图识别结果
            steps: 决策步骤列表
            
        Returns:
            tuple: (最终响应, 工具链执行结果列表)
        """
        toolchain_results: List[ToolchainExecutionResult] = []
        
        # 创建工具链执行步骤
        toolchain_exec_step = DecisionStep(
            step_name="toolchain_execution",
            step_description=f"开始执行工具链（共{len(toolchain_plan)}步）",
            status="running",
            sub_steps=[]
        )
        steps.append(toolchain_exec_step)
        
        # 用于存储中间结果，供后续步骤使用
        intermediate_results = {}
        
        for i, tc_step in enumerate(toolchain_plan, 1):
            step_start_time = time.time()
            module_name = tc_step.module
            action = tc_step.action
            
            # 创建子步骤记录
            tc_sub_step = DecisionStep(
                step_name=f"toolchain_exec_{i}",
                step_description=f"步骤{i}: 执行 {module_name}.{action}",
                status="running"
            )
            toolchain_exec_step.sub_steps.append(tc_sub_step)
            
            logger.info(f"执行工具链步骤 {i}/{len(toolchain_plan)}: {module_name}.{action}")
            
            # 获取模块
            module = self.module_registry.get_module(module_name)
            
            if module is None:
                duration = (time.time() - step_start_time) * 1000
                error_msg = f"模块 {module_name} 未找到"
                
                tc_sub_step.status = "failed"
                tc_sub_step.success = False
                tc_sub_step.duration_ms = duration
                tc_sub_step.result = {"error": error_msg}
                
                toolchain_results.append(ToolchainExecutionResult(
                    step_number=i,
                    module_name=module_name,
                    action=action,
                    success=False,
                    result=None,
                    message=error_msg,
                    duration_ms=duration
                ))
                
                # 如果某一步失败，终止工具链
                toolchain_exec_step.status = "failed"
                toolchain_exec_step.success = False
                toolchain_exec_step.step_description = f"工具链执行失败：步骤{i}失败"
                
                error_response = f"工具链执行失败：步骤{i}（{module_name}模块）无法执行。\n错误：{error_msg}"
                return error_response, toolchain_results
            
            # 准备执行上下文
            exec_context = {
                "history": context,
                "session_id": session_id,
                "intent": intent_result.intent_type.value,
                "action": action,
                "slots": {k: v.value for k, v in intent_result.slots.items()},
                "toolchain_step": i,
                "toolchain_total": len(toolchain_plan),
                "intermediate_results": intermediate_results,
                "original_query": user_input
            }
            
            try:
                # 执行模块
                module_result = module.execute(
                    query=user_input,
                    context=exec_context
                )
                duration = (time.time() - step_start_time) * 1000
                
                # 更新子步骤状态
                tc_sub_step.status = "completed"
                tc_sub_step.success = module_result.success
                tc_sub_step.duration_ms = duration
                tc_sub_step.result = module_result.to_dict()
                tc_sub_step.step_description = f"步骤{i}: {module_name}.{action} - {'成功' if module_result.success else '失败'}"
                
                # 保存结果
                result_data = module_result.data if module_result.success else module_result.message
                toolchain_results.append(ToolchainExecutionResult(
                    step_number=i,
                    module_name=module_name,
                    action=action,
                    success=module_result.success,
                    result=result_data,
                    message=module_result.message,
                    duration_ms=duration
                ))
                
                # 保存中间结果供后续步骤使用
                intermediate_results[f"step_{i}"] = {
                    "module": module_name,
                    "action": action,
                    "result": result_data,
                    "success": module_result.success
                }
                
                # 如果某一步失败，终止工具链
                if not module_result.success:
                    toolchain_exec_step.status = "failed"
                    toolchain_exec_step.success = False
                    toolchain_exec_step.step_description = f"工具链执行失败：步骤{i}执行失败"
                    
                    error_response = f"工具链执行失败：步骤{i}（{module_name}模块）执行失败。\n原因：{module_result.message}"
                    return error_response, toolchain_results
                
            except Exception as e:
                duration = (time.time() - step_start_time) * 1000
                error_msg = str(e)
                
                tc_sub_step.status = "failed"
                tc_sub_step.success = False
                tc_sub_step.duration_ms = duration
                tc_sub_step.result = {"error": error_msg}
                
                toolchain_results.append(ToolchainExecutionResult(
                    step_number=i,
                    module_name=module_name,
                    action=action,
                    success=False,
                    result=None,
                    message=f"执行异常: {error_msg}",
                    duration_ms=duration
                ))
                
                toolchain_exec_step.status = "failed"
                toolchain_exec_step.success = False
                toolchain_exec_step.step_description = f"工具链执行失败：步骤{i}发生异常"
                
                error_response = f"工具链执行失败：步骤{i}（{module_name}模块）发生异常。\n错误：{error_msg}"
                return error_response, toolchain_results
        
        # 工具链执行完成
        toolchain_exec_step.status = "completed"
        toolchain_exec_step.success = True
        toolchain_exec_step.step_description = f"工具链执行完成：共{len(toolchain_plan)}步全部成功"
        
        # 生成综合响应
        final_response = self._generate_toolchain_response(toolchain_results, user_input)
        
        return final_response, toolchain_results
    
    def _generate_toolchain_response(
        self,
        toolchain_results: List[ToolchainExecutionResult],
        original_query: str
    ) -> str:
        """
        生成工具链执行的综合响应
        
        Args:
            toolchain_results: 工具链执行结果列表
            original_query: 原始用户查询
            
        Returns:
            str: 综合响应文本
        """
        response_parts = ["🔗 工具链执行完成\n"]
        response_parts.append(f"共执行 {len(toolchain_results)} 个步骤：\n")
        
        for result in toolchain_results:
            status_icon = "✅" if result.success else "❌"
            response_parts.append(f"\n{status_icon} 步骤{result.step_number}: {result.module_name} 模块")
            response_parts.append(f"   动作: {result.action}")
            if isinstance(result.result, str):
                # 提取结果的关键信息（简化显示）
                result_summary = result.result.split('\n')[0] if result.result else "完成"
                response_parts.append(f"   结果: {result_summary}")
        
        response_parts.append("\n" + "="*40)
        
        # 最后一步的结果作为主要输出
        if toolchain_results:
            final_result = toolchain_results[-1]
            response_parts.append(f"\n📋 最终结果：")
            if isinstance(final_result.result, str):
                response_parts.append(final_result.result)
            else:
                response_parts.append(str(final_result.result))
        
        return "\n".join(response_parts)
    
    def get_status(self) -> Dict:
        """获取决策中枢状态"""
        return {
            "initialized": self._initialized,
            "intent_threshold": self.intent_threshold,
            "max_context_turns": self.max_context_turns,
            "has_model_client": self.model_client is not None,
            "registered_modules": [m.to_dict() for m in self.module_registry.list_modules()],
            "active_sessions": len(self.context_manager.get_all_sessions())
        }
    
    def clear_session(self, session_id: str):
        """清除会话"""
        self.context_manager.clear_context(session_id)
    
    def shutdown(self):
        """关闭决策中枢"""
        logger.info("关闭智能决策中枢...")
        self.module_registry.shutdown_all()
        self._initialized = False
