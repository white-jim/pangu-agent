"""
Gradio Web演示界面
增强版：展示更多系统运行细节
"""
import gradio as gr
from typing import Generator, Dict, List, Optional, Any
import logging
import html

logger = logging.getLogger(__name__)


class GradioInterface:
    """
    Gradio Web界面类
    提供用户交互界面
    """
    
    def __init__(self, agent: Any):
        """
        初始化界面
        
        Args:
            agent: 智能体实例
        """
        self.agent = agent
        self.demo: Optional[gr.Blocks] = None
        self.session_id = "gradio_session"
    
    def build_interface(self) -> gr.Blocks:
        """构建Gradio界面"""
        with gr.Blocks(
            title="基于OpenPangu大模型的问答智能体",
            css="""
            .thinking-box {background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin: 10px 0; border: 1px solid #e9ecef;}
            .step-item {margin: 8px 0; padding: 10px; border-left: 3px solid #4a90d9; background-color: #fff; border-radius: 0 5px 5px 0;}
            .step-running {border-left-color: #ffc107; background-color: #fffbf0;}
            .step-success {border-left-color: #28a745;}
            .step-failed {border-left-color: #dc3545;}
            .sub-step {margin: 5px 0 5px 20px; padding: 8px; border-left: 2px solid #dee2e6; background-color: #fafafa; font-size: 0.9em;}
            .intent-badge {display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.85em; margin: 2px;}
            .intent-qa {background-color: #e3f2fd; color: #1565c0;}
            .intent-navigation {background-color: #e8f5e9; color: #2e7d32;}
            .intent-vision {background-color: #fff3e0; color: #ef6c00;}
            .intent-registration {background-color: #f3e5f5; color: #7b1fa2;}
            .intent-system {background-color: #eceff1; color: #546e7a;}
            .confidence-bar {height: 8px; background: linear-gradient(90deg, #4a90d9, #28a745); border-radius: 4px;}
            .rag-box {background-color: #f5fff5; border-radius: 10px; padding: 15px; margin: 10px 0; border: 1px solid #e8f5e9;}
            .response-box {background-color: #fff8f0; border-radius: 10px; padding: 15px; margin: 10px 0; border: 1px solid #fff3e0;}
            .metadata-box {background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin: 10px 0; border: 1px solid #e9ecef; font-size: 0.9em;}
            .status-panel {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; padding: 15px;}
            .module-status {display: inline-block; padding: 2px 8px; border-radius: 3px; margin: 2px; font-size: 0.8em;}
            .module-active {background-color: #4caf50; color: white;}
            .module-inactive {background-color: #9e9e9e; color: white;}
            .tool-call-box {background-color: #f0f7ff; border-radius: 10px; padding: 15px; margin: 10px 0; border: 1px solid #c8e6c9;}
            """
        ) as demo:
            gr.Markdown(
                """
                # 🤖 基于OpenPangu大模型的问答智能体
                
                本系统采用"智能决策中枢 + 分布式功能模块"架构，集成大模型进行意图识别和动作决策。
                
                **核心能力：**
                - 🧠 大模型意图识别（意图+动作联合识别）
                - 📚 RAG知识检索增强
                - 🗺️ 导航模块（展示地图、导航带路、位置查询）
                - 👁️ 视觉模块（人脸检测、识别、注册、场景分析）
                - 💬 多轮对话上下文管理
                - 🔗 工具链调用（多模块协同）
                """
            )
            
            with gr.Row(equal_height=False):
                # 左侧：输入区域（保持原有位置）
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="请输入您的问题",
                        placeholder="例如：\n- 这个项目是什么？\n- 带我去会议室A\n- 看看这是谁\n- 展示地图\n- 帮我注册张三的人脸\n- 去查看一下一号安全出口是否畅通",
                        lines=4,
                        max_lines=6
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("🚀 提交问题", variant="primary", scale=2)
                        clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", scale=1)
                    
                    with gr.Accordion("📊 系统状态", open=True):
                        status_display = gr.HTML(
                            label="系统状态",
                            value=self._get_default_status_html()
                        )
                        with gr.Row():
                            status_btn = gr.Button("🔄 刷新状态", variant="secondary")
                
                # 右侧：展示区域（上移，不与输入框底部对齐）
                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.Tab("🧠 推理过程"):
                            thinking_output = gr.HTML(
                                label="推理步骤",
                                value="<div class='thinking-box'>等待用户提问...</div>"
                            )
                        
                        with gr.Tab("📚 RAG检索"):
                            rag_output = gr.HTML(
                                label="检索到的相关知识",
                                value="<div class='rag-box'>暂无检索结果</div>"
                            )
                        
                        with gr.Tab("🔧 工具调用"):
                            tool_call_output = gr.HTML(
                                label="工具调用详情",
                                value="<div class='tool-call-box'>暂无工具调用</div>"
                            )
                        
                        with gr.Tab("💬 回答结果"):
                            response_output = gr.Markdown(
                                label="智能体回答",
                                value="等待生成回答..."
                            )
                        
                        with gr.Tab("📊 执行详情"):
                            with gr.Accordion("意图识别详情", open=True):
                                intent_detail_output = gr.HTML(
                                    label="意图识别详细信息",
                                    value="<div class='metadata-box'>等待识别...</div>"
                                )
                            with gr.Accordion("模块执行详情", open=True):
                                module_detail_output = gr.HTML(
                                    label="模块执行详情",
                                    value="<div class='metadata-box'>等待执行...</div>"
                                )
            
            def process_question(question: str) -> tuple:
                """处理问题并返回各阶段输出"""
                if not question.strip():
                    return (
                        "<div class='thinking-box'>请输入问题后点击提交</div>",
                        "<div class='rag-box'>暂无检索结果</div>",
                        "<div class='tool-call-box'>暂无工具调用</div>",
                        "请先提问",
                        "<div class='metadata-box'>等待识别...</div>",
                        "<div class='metadata-box'>等待执行...</div>"
                    )
                
                thinking_html = "<div class='thinking-box'>"
                thinking_html += "<h4>🧠 智能体思考过程</h4>"
                thinking_html += "<div class='step-item step-running'>⏳ <b>步骤1:</b> 预处理用户输入...</div>"
                yield (
                    thinking_html + "</div>",
                    "<div class='rag-box'>正在处理...</div>",
                    "<div class='tool-call-box'>正在分析...</div>",
                    "等待处理...",
                    "<div class='metadata-box'>正在识别...</div>",
                    "<div class='metadata-box'>等待执行...</div>"
                )
                
                result = self.agent.process(question, self.session_id)
                
                steps = result.get("steps", [])
                is_toolchain = result.get("is_toolchain_execution", False)
                toolchain_results = result.get("toolchain_results", [])
                
                thinking_html = "<div class='thinking-box'>"
                thinking_html += "<h4>🧠 智能体思考过程</h4>"
                
                # 显示是否为工具链执行
                if is_toolchain:
                    thinking_html += f"<div class='step-item step-running' style='background-color:#e3f2fd;'>"
                    thinking_html += f"🔗 <b>检测到工具链调用</b><br/>"
                    thinking_html += f"<small>需要按顺序执行 {len(toolchain_results)} 个模块</small>"
                    thinking_html += "</div>"
                
                for i, step in enumerate(steps, 1):
                    status_icon = "✅" if step.get("success", True) else "❌"
                    status_class = "step-success" if step.get("success", True) else "step-failed"
                    if step.get("status") == "running":
                        status_icon = "🔄"
                        status_class = "step-running"
                    
                    step_name = step.get('step_name', '')
                    
                    # 工具链执行步骤特殊标记
                    if step_name == "toolchain_execution":
                        status_icon = "🔗"
                        status_class = "step-success" if step.get("success", True) else "step-failed"
                    
                    thinking_html += f"<div class='step-item {status_class}'>{status_icon} <b>步骤{i}: {step_name}</b><br/>"
                    thinking_html += f"<small>{step.get('step_description', '')}</small>"
                    
                    if step.get("duration_ms", 0) > 0:
                        thinking_html += f"<br/><small style='color:#666;'>⏱️ 耗时: {step.get('duration_ms', 0):.2f}ms</small>"
                    
                    sub_steps = step.get("sub_steps", [])
                    for sub_step in sub_steps:
                        sub_status = "✅" if sub_step.get("success", True) else "❌"
                        sub_desc = sub_step.get('step_description', '')
                        
                        # 工具链子步骤特殊样式
                        if 'toolchain' in sub_step.get('step_name', ''):
                            thinking_html += f"<div class='sub-step' style='border-left-color:#4a90d9;background-color:#f0f7ff;'>{sub_status} {sub_desc}"
                        else:
                            thinking_html += f"<div class='sub-step'>{sub_status} {sub_desc}"
                        
                        if sub_step.get("duration_ms", 0) > 0:
                            thinking_html += f" <span style='color:#888;'>({sub_step.get('duration_ms', 0):.2f}ms)</span>"
                        thinking_html += "</div>"
                    
                    thinking_html += "</div>"
                
                # 添加工具链执行摘要
                if is_toolchain and toolchain_results:
                    thinking_html += "<div style='margin-top:15px;padding:10px;background:#f5f5f5;border-radius:8px;'>"
                    thinking_html += "<b>🔗 工具链执行摘要</b><br/>"
                    for tc_result in toolchain_results:
                        tc_icon = "✅" if tc_result.get("success") else "❌"
                        tc_module = tc_result.get("module_name", "未知")
                        tc_action = tc_result.get("action", "未知")
                        tc_duration = tc_result.get("duration_ms", 0)
                        thinking_html += f"<div style='margin:5px 0;padding:5px 10px;background:white;border-radius:4px;'>"
                        thinking_html += f"{tc_icon} 步骤{tc_result.get('step_number')}: {tc_module}.{tc_action} ({tc_duration:.0f}ms)"
                        thinking_html += "</div>"
                    thinking_html += "</div>"
                
                thinking_html += "</div>"
                
                rag_html = "<div class='rag-box'>"
                rag_html += "<h4>📚 RAG检索结果</h4>"
                
                metadata = result.get("metadata", {})
                retrieved_docs = metadata.get("retrieved_docs", [])
                
                if retrieved_docs:
                    rag_html += f"<p>找到 <b>{len(retrieved_docs)}</b> 条相关知识：</p>"
                    for i, doc in enumerate(retrieved_docs, 1):
                        score = doc.get("score", 0)
                        content = doc.get("content", "")
                        source = doc.get("metadata", {}).get("source", "未知来源")
                        rag_html += f"""
                        <div style='margin: 10px 0; padding: 12px; background: white; border-radius: 6px; border-left: 3px solid #4CAF50;'>
                            <b>[文档{i}]</b> <small style='color:#888;'>相关度: {score:.3f} | 来源: {source}</small><br/>
                            {content[:300]}{'...' if len(content) > 300 else ''}
                        </div>
                        """
                else:
                    rag_html += "<p>本次问答未使用知识库检索或未找到相关文档</p>"
                
                rag_html += "</div>"
                
                tool_call_html = "<div class='tool-call-box'>"
                tool_call_html += "<h4>🔧 工具调用详情</h4>"
                
                intent = result.get("intent", {})
                module_name = result.get("module_name", "未知")
                action_taken = result.get("action_taken", "未知")
                intent_description = intent.get("intent_description", "未知")
                action_description = intent.get("action_description", "未知")
                reasoning = intent.get("reasoning", "")
                
                needs_toolchain = metadata.get("needs_toolchain", False)
                toolchain_plan = metadata.get("toolchain_plan", [])
                toolchain_results = result.get("toolchain_results", [])
                is_toolchain = result.get("is_toolchain_execution", False)
                
                if is_toolchain:
                    # 工具链调用展示
                    tool_call_html += f"<div style='background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:12px 15px;border-radius:8px;margin-bottom:15px;'>"
                    tool_call_html += f"<b>🔗 工具链调用模式</b><br/>"
                    tool_call_html += f"<small>共 {len(toolchain_results)} 个步骤按顺序执行</small>"
                    tool_call_html += "</div>"
                    
                    # 工具链规划
                    if toolchain_plan:
                        tool_call_html += f"<p><b>📋 工具链规划:</b></p>"
                        tool_call_html += "<div style='background:#f8f9fa;padding:10px;border-radius:6px;margin:10px 0;'>"
                        for step in toolchain_plan:
                            step_num = step.get("step", 0)
                            step_module = step.get("module", "")
                            step_action = step.get("action", "")
                            step_desc = step.get("description", "")
                            tool_call_html += f"<div style='margin:8px 0;padding:8px 12px;background:white;border-radius:4px;border-left:3px solid #4a90d9;'>"
                            tool_call_html += f"<b>步骤{step_num}:</b> {step_module} 模块<br/>"
                            tool_call_html += f"<small style='color:#666;'>动作: {step_action} | {step_desc}</small>"
                            tool_call_html += "</div>"
                        tool_call_html += "</div>"
                    
                    # 工具链实际执行结果
                    if toolchain_results:
                        tool_call_html += f"<p><b>⚙️ 实际执行结果:</b></p>"
                        for tc_result in toolchain_results:
                            step_num = tc_result.get("step_number", 0)
                            tc_module = tc_result.get("module_name", "未知")
                            tc_action = tc_result.get("action", "未知")
                            tc_success = tc_result.get("success", False)
                            tc_duration = tc_result.get("duration_ms", 0)
                            tc_message = tc_result.get("message", "")
                            
                            status_color = "#28a745" if tc_success else "#dc3545"
                            status_bg = "#f0fff0" if tc_success else "#fff5f5"
                            status_icon = "✅" if tc_success else "❌"
                            
                            tool_call_html += f"<div style='margin:10px 0;padding:12px;background:{status_bg};border-radius:6px;border:1px solid {status_color};'>"
                            tool_call_html += f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                            tool_call_html += f"<span><b>{status_icon} 步骤{step_num}: {tc_module} 模块</b></span>"
                            tool_call_html += f"<span style='color:#666;font-size:0.85em;'>⏱️ {tc_duration:.0f}ms</span>"
                            tool_call_html += "</div>"
                            tool_call_html += f"<div style='margin-top:5px;color:#555;'><small>动作: {tc_action}</small></div>"
                            if tc_message:
                                tool_call_html += f"<div style='margin-top:5px;color:#666;font-size:0.9em;'><small>{tc_message}</small></div>"
                            tool_call_html += "</div>"
                    
                    # 工具链执行总结
                    all_success = metadata.get("toolchain_all_success", False)
                    completed_count = metadata.get("toolchain_completed_count", 0)
                    total_count = metadata.get("toolchain_step_count", 0)
                    
                    tool_call_html += f"<div style='margin-top:15px;padding:12px;background:{'#f0fff0' if all_success else '#fff5f5'};border-radius:6px;text-align:center;'>"
                    tool_call_html += f"<b>{'✅ 工具链执行成功' if all_success else '⚠️ 工具链执行未完成'}</b><br/>"
                    tool_call_html += f"<small>完成进度: {completed_count}/{total_count} 步骤</small>"
                    tool_call_html += "</div>"
                    
                else:
                    # 单模块调用展示
                    tool_call_html += f"<p><span style='background:#e3f2fd;color:#1565c0;padding:4px 8px;border-radius:4px;'>🔧 单模块调用</span></p>"
                    
                    if needs_toolchain and toolchain_plan:
                        tool_call_html += f"<p><b>📋 工具链规划（未触发）:</b></p>"
                        tool_call_html += "<ol>"
                        for step in toolchain_plan:
                            step_num = step.get("step", 0)
                            step_module = step.get("module", "")
                            step_action = step.get("action", "")
                            step_desc = step.get("description", "")
                            tool_call_html += f"<li><b>步骤{step_num}:</b> {step_module} 模块 - {step_desc} <small>(动作: {step_action})</small></li>"
                        tool_call_html += "</ol>"
                        tool_call_html += "<hr/>"
                    
                    tool_call_html += f"<p><b>调用工具:</b> {module_name} 模块</p>"
                    tool_call_html += f"<p><b>执行动作:</b> {action_description} ({action_taken})</p>"
                    tool_call_html += f"<p><b>识别意图:</b> {intent_description}</p>"
                    tool_call_html += f"<p><b>调用原因:</b> {reasoning if reasoning else '根据用户输入自动识别并调用相应工具'}</p>"
                    
                    tool_call_html += "<hr/><p><b>决策依据:</b></p><ul>"
                    tool_call_html += f"<li>用户意图分析: {intent_description}</li>"
                    tool_call_html += f"<li>任务类型: {action_description}</li>"
                    tool_call_html += f"<li>处理模块: {module_name}</li>"
                    tool_call_html += "</ul>"
                    
                    tool_call_html += "<p><b>执行结果:</b></p>"
                    tool_call_html += "<ul>"
                    tool_call_html += f"<li>模块调用状态: {'✅ 成功' if result.get('success') else '❌ 失败'}</li>"
                    tool_call_html += f"<li>返回数据: 已生成</li>"
                    tool_call_html += "</ul>"
                
                tool_call_html += "</div>"
                
                response_content = result.get("response", "")
                
                intent_html = "<div class='metadata-box'>"
                intent_html += "<h4>🎯 意图识别详情</h4>"
                
                intent_type = intent.get("intent_type", "unknown")
                action_type = intent.get("action_type", "unknown")
                confidence = intent.get("confidence", 0)
                method = intent.get("recognition_method", "unknown")
                
                intent_class = f"intent-{intent_type}"
                intent_html += f"<p><span class='intent-badge {intent_class}'>{intent.get('intent_description', intent_type)}</span>"
                intent_html += f" <span class='intent-badge' style='background:#fff3e0;'>{intent.get('action_description', action_type)}</span></p>"
                
                intent_html += f"<p><b>置信度:</b> {confidence:.2f} "
                confidence_percent = int(confidence * 100)
                intent_html += f"<div style='background:#e0e0e0;height:10px;border-radius:5px;width:150px;display:inline-block;vertical-align:middle;margin-left:10px;'><div style='background:linear-gradient(90deg,#4a90d9,#28a745);height:100%;width:{confidence_percent}%;border-radius:5px;'></div></div> ({confidence_percent}%)</p>"
                
                intent_html += f"<p><b>识别方式:</b> {'🔮 大模型推理' if method == 'llm' else '📋 规则匹配'}</p>"
                
                if reasoning:
                    intent_html += f"<p><b>推理过程:</b> {reasoning}</p>"
                
                slots = intent.get("slots", {})
                if slots:
                    intent_html += "<p><b>提取的槽位:</b></p><ul>"
                    for slot_name, slot_info in slots.items():
                        intent_html += f"<li><b>{slot_name}:</b> {slot_info.get('value', 'N/A')}</li>"
                    intent_html += "</ul>"
                
                intent_html += f"<p><b>大模型原始输出:</b></p>"
                llm_resp = intent.get('llm_response', 'N/A')
                if llm_resp:
                    llm_resp_display = llm_resp[:800] + ("..." if len(llm_resp) > 800 else "")
                    intent_html += f"<pre style='background:#f5f5f5;padding:10px;border-radius:5px;overflow-x:auto;font-size:0.85em;'>{html.escape(llm_resp_display)}</pre>"
                else:
                    intent_html += "<p>无</p>"
                
                intent_html += "</div>"
                
                module_html = "<div class='metadata-box'>"
                module_html += "<h4>⚙️ 模块执行详情</h4>"
                
                is_toolchain = result.get("is_toolchain_execution", False)
                toolchain_results = result.get("toolchain_results", [])
                
                if is_toolchain:
                    # 工具链执行详情
                    module_html += f"<p><span style='background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:4px 10px;border-radius:4px;'>🔗 工具链模式</span></p>"
                    module_html += f"<p><b>执行模块数:</b> {len(toolchain_results)} 个</p>"
                    
                    if toolchain_results:
                        module_html += "<p><b>各模块执行详情:</b></p>"
                        for tc_result in toolchain_results:
                            step_num = tc_result.get("step_number", 0)
                            tc_module = tc_result.get("module_name", "未知")
                            tc_action = tc_result.get("action", "未知")
                            tc_success = tc_result.get("success", False)
                            tc_duration = tc_result.get("duration_ms", 0)
                            
                            status_color = "#28a745" if tc_success else "#dc3545"
                            module_html += f"<div style='margin:8px 0;padding:10px;border-left:3px solid {status_color};background:#fafafa;border-radius:0 5px 5px 0;'>"
                            module_html += f"<b>步骤{step_num}: {tc_module}</b> ({tc_action})<br/>"
                            module_html += f"<small>状态: {'成功' if tc_success else '失败'} | 耗时: {tc_duration:.2f}ms</small>"
                            module_html += "</div>"
                    
                    all_success = metadata.get("toolchain_all_success", False)
                    module_html += f"<p><b>整体执行状态:</b> {'✅ 全部成功' if all_success else '❌ 部分失败'}</p>"
                else:
                    # 单模块执行详情
                    module_html += f"<p><b>调用的模块:</b> {module_name}</p>"
                    module_html += f"<p><b>执行的动作:</b> {action_taken}</p>"
                    
                    exec_metadata = metadata or {}
                    if exec_metadata:
                        module_html += "<p><b>执行结果:</b></p><ul>"
                        for key, value in exec_metadata.items():
                            if key != "retrieved_docs" and not key.startswith("toolchain"):
                                module_html += f"<li><b>{key}:</b> {str(value)[:150]}</li>"
                        module_html += "</ul>"
                
                # 通用执行统计
                total_time = sum(s.get("duration_ms", 0) for s in steps)
                module_html += f"<p><b>总执行时间:</b> {total_time:.2f}ms</p>"
                
                module_html += "<p><b>各步骤耗时:</b></p><ul>"
                for step in steps:
                    step_name = step.get('step_name', '')
                    step_duration = step.get('duration_ms', 0)
                    # 工具链执行步骤特殊标记
                    if step_name == "toolchain_execution":
                        module_html += f"<li>🔗 {step_name}: {step_duration:.2f}ms <span style='color:#667eea;'>(工具链)</span></li>"
                    else:
                        module_html += f"<li>{step_name}: {step_duration:.2f}ms</li>"
                module_html += "</ul>"
                
                module_html += "</div>"
                
                yield (thinking_html, rag_html, tool_call_html, response_content, intent_html, module_html)
            
            def clear_conversation():
                """清空对话"""
                self.agent.clear_session(self.session_id)
                return (
                    "<div class='thinking-box'>对话已清空，等待新的提问...</div>",
                    "<div class='rag-box'>暂无检索结果</div>",
                    "<div class='tool-call-box'>暂无工具调用</div>",
                    "等待生成回答...",
                    "<div class='metadata-box'>等待识别...</div>",
                    "<div class='metadata-box'>等待执行...</div>"
                )
            
            def refresh_status_display():
                """刷新状态面板显示"""
                try:
                    status = self.agent.get_status()
                    return self._build_status_html(status)
                except Exception as e:
                    return f"<div style='color:red;'>获取状态失败: {str(e)}</div>"
            
            submit_btn.click(
                fn=process_question,
                inputs=[question_input],
                outputs=[thinking_output, rag_output, tool_call_output, response_output, intent_detail_output, module_detail_output]
            )
            
            clear_btn.click(
                fn=clear_conversation,
                outputs=[thinking_output, rag_output, tool_call_output, response_output, intent_detail_output, module_detail_output]
            )
            
            status_btn.click(
                fn=refresh_status_display,
                outputs=[status_display]
            )
            
            gr.Markdown(
                """
                ---
                <div style='text-align: center; color: #666;'>
                    <p>基于OpenPangu大模型的问答智能体 | 毕业设计</p>
                    <p>技术栈: Python + Transformers + FAISS + Gradio + DeepSeek-V3.2</p>
                </div>
                """
            )
        
        self.demo = demo
        return demo
    
    def _get_default_status_html(self) -> str:
        """获取默认状态面板HTML"""
        return """
        <div class='status-panel'>
            <h4>📊 系统状态</h4>
            <p>点击"刷新状态"按钮查看详细信息</p>
        </div>
        """
    
    def _build_status_html(self, status: Dict) -> str:
        """构建状态面板HTML"""
        html = """
        <div class='status-panel'>
            <h4>📊 系统运行状态</h4>
        """
        
        html += f"<p><b>初始化状态:</b> {'✅ 已初始化' if status.get('initialized') else '❌ 未初始化'}</p>"
        
        modules = status.get("registered_modules", [])
        if modules:
            html += "<p><b>模块状态:</b></p>"
            html += "<div style='margin-left:15px;'>"
            for module in modules:
                name = module.get("name", "unknown")
                initialized = module.get("initialized", False)
                html += f"<span class='module-status {'module-active' if initialized else 'module-inactive'}'>{name}</span> "
            html += "</div>"
        
        html += f"<p><b>意图识别阈值:</b> {status.get('intent_threshold', '0.6')}</p>"
        html += f"<p><b>最大上下文轮数:</b> {status.get('max_context_turns', '5')}</p>"
        html += f"<p><b>模型客户端:</b> {'✅ 已配置' if status.get('has_model_client') else '❌ 未配置'}</p>"
        html += f"<p><b>活跃会话数:</b> {status.get('active_sessions', 0)}</p>"
        
        html += "</div>"
        return html
    
    def launch(
        self,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False,
        **kwargs
    ):
        """
        启动Gradio服务
        
        Args:
            server_name: 服务器地址
            server_port: 端口号
            share: 是否创建公网链接
            **kwargs: 其他参数
        """
        if self.demo is None:
            self.build_interface()
        
        logger.info(f"启动Gradio服务: http://{server_name}:{server_port}")
        self.demo.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            **kwargs
        )
