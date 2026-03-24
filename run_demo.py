"""
演示脚本
不依赖大模型，仅使用RAG检索功能进行演示
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pangu_agent import PanguAgent
from pangu_agent.web import GradioInterface


def main():
    print("=" * 60)
    print("基于OpenPangu大模型的问答智能体 - 中期演示")
    print("（演示模式：使用RAG检索，无大模型生成）")
    print("=" * 60)
    
    agent = PanguAgent()
    
    print("\n正在初始化智能体...")
    agent.initialize()
    
    print("\n正在启动Web界面...")
    interface = GradioInterface(agent)
    
    print("\n" + "=" * 60)
    print("🚀 服务启动成功！")
    print("📍 本地访问地址: http://localhost:7860")
    print("=" * 60 + "\n")
    
    interface.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
