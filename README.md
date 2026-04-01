# 基于OpenPangu大模型的问答智能体

## 项目简介

本项目是一个基于OpenPangu大模型的问答智能体系统，采用"智能决策中枢 + 分布式功能模块"架构设计。OpenPangu大模型作为核心大脑，负责理解用户意图、调度各个功能模块、生成最终回答。

### 核心特性

- 🎯 **智能意图识别**：自动判断用户问题类型（问答/导航/视觉/注册等）
- 📚 **RAG知识检索**：基于向量数据库的知识检索增强生成
- 💬 **多轮对话**：支持上下文记忆的多轮对话交互
- 🔧 **模块化架构**：分布式功能模块，易于扩展
- 🖥️ **Web演示界面**：基于Gradio的可视化交互界面

## 项目结构

```
pangu_agent/
├── config/                     # 配置模块
│   ├── __init__.py
│   └── settings.py             # 配置定义
├── models/                     # 模型模块
│   ├── __init__.py
│   ├── pangu_model.py          # OpenPangu模型加载与推理
│   └── external_api.py         # 外部大模型API接口
├── decision_center/            # 智能决策中枢
│   ├── __init__.py
│   ├── decision_center.py      # 决策中枢核心
│   ├── intent_recognizer.py    # 意图识别
│   ├── module_registry.py      # 模块注册发现
│   └── context_manager.py      # 上下文管理
├── modules/                    # 功能模块
│   ├── semantic_interaction/   # 通用语义问答（RAG）
│   │   ├── __init__.py
│   │   ├── module.py
│   │   ├── rag_engine.py
│   │   └── vector_store.py
│   ├── navigation/             # 导航导览（框架）
│   ├── vision/                 # 视觉交互（框架）
│   ├── interaction/            # 信息展示与语音
│   └── web_admin/              # Web后台管理（框架）
├── web/                        # Web界面
│   ├── __init__.py
│   └── gradio_interface.py
├── agent.py                    # 智能体主类
└── __init__.py
├── main.py                     # 主入口
├── run_demo.py                 # 演示脚本
├── requirements.txt            # 依赖清单
└── README.md                   # 项目说明
```

## 环境要求

- Python 3.8+
- CUDA 11.x（GPU推理）或 昇腾NPU环境

## 安装步骤

### 1. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置模型路径（可选）

设置OpenPangu模型路径环境变量：

```bash
# Linux/Mac
export PANGU_MODEL_PATH=/path/to/pangu/model

# Windows
set PANGU_MODEL_PATH=D:\models\pangu
```

## 运行方法

### 演示模式（无需大模型）

使用RAG检索功能进行演示，无需加载大模型：

```bash
python run_demo.py
```

### 完整模式（需要大模型）

```bash
# 使用本地OpenPangu模型
python main.py --model-path /path/to/pangu/model

# 使用外部API
python main.py --use-external-api --api-key YOUR_API_KEY
```

### 启动参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-path` | OpenPangu模型路径 | 环境变量PANGU_MODEL_PATH |
| `--use-external-api` | 使用外部API | False |
| `--api-key` | 外部API密钥 | 环境变量EXTERNAL_API_KEY |
| `--api-base` | 外部API地址 | 环境变量EXTERNAL_API_BASE |
| `--port` | Gradio服务端口 | 7860 |
| `--host` | Gradio服务地址 | 0.0.0.0 |
| `--share` | 创建公网分享链接 | False |
| `--log-level` | 日志级别 | INFO |

## 模块说明

### 1. 智能决策中枢

核心调度模块，负责：
- **意图识别**：基于关键词和规则判断用户意图类型
- **模块调度**：根据意图类型调用对应功能模块
- **上下文管理**：维护多轮对话历史
- **结果整合**：整合各模块结果生成最终回答

### 2. 通用语义交互模块（RAG）

基于检索增强生成的问答模块：
- 向量数据库存储知识
- 语义相似度检索
- 结合大模型生成回答

### 3. 导航导览模块（框架）

预留接口，待实现：
- 室内地图管理
- 路径规划
- 实时导航

### 4. 视觉交互模块（框架）

预留接口，待实现：
- 人脸检测
- 人脸识别
- 人脸注册

### 5. 信息展示与语音模块

基础框架实现：
- 文本展示
- 语音合成接口（待实现）
- 语音识别接口（待实现）

## 核心链路

```
用户提问 → 意图识别 → 模块调度 → RAG检索 → 大模型生成 → 输出回答
```

## 示例问答

系统内置了关于项目本身的示例知识，可以尝试以下问题：

- 这个项目是做什么的？
- OpenPangu大模型是什么？
- 智能决策中枢有什么功能？
- RAG技术是什么？
- 系统支持哪些功能模块？

## 昇腾NPU配置

在昇腾NPU环境下运行需要：

1. 安装昇腾CANN工具包
2. 安装torch-npu
3. 配置环境变量：

```bash
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit
```

## 已知问题与待优化

### 1. 工具链执行容错与重新规划

**当前问题**：
- 工具链执行采用简单的"失败即终止"策略
- 某一步失败后直接终止整个流程，没有重试机制
- 没有备用执行路径
- 没有结合已完成步骤和意图识别结果进行重新规划

**优化方向**：
- 添加步骤重试机制
- 支持备用执行路径
- 实现动态重新规划：根据已完成步骤和当前状态调整后续流程
- 支持部分成功场景，返回已完成的结果

### 2. 语义交互模块输入信息缺失

**当前问题**：
- RAG引擎输入给大模型的内容仅包含：用户问题 + RAG检索结果
- 缺少历史对话上下文（尽管ContextManager中有保存，但未传递给RAG引擎）
- 工具链执行的中间结果未传递给大模型用于生成最终回答
- 前面智能体执行命令的结果未被利用

**优化方向**：
- 将历史对话上下文整合到RAG提示词中
- 将工具链执行的中间结果传递给大模型
- 让大模型能够基于完整的执行历史生成更准确的回答

## 开发计划

- [ ] 完善导航导览模块
- [ ] 实现视觉交互功能
- [ ] 添加语音交互支持
- [ ] 完善Web后台管理
- [ ] 支持更多大模型

## 许可证

MIT License

## 作者

毕业设计项目 - 基于OpenPangu大模型的问答智能体设计与实现
