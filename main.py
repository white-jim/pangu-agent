"""
主入口文件
启动Gradio Web演示界面
"""
import os
import sys
import logging
import argparse
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 加载环境变量配置
from pangu_agent.config import get_settings
settings = get_settings()

from pangu_agent import PanguAgent
from pangu_agent.web import GradioInterface


def setup_logging(log_level: str = "INFO"):
    """配置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("pangu_agent.log", encoding="utf-8")
        ]
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="基于OpenPangu大模型的问答智能体 - 中期演示"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("PANGU_MODEL_PATH", ""),
        help="OpenPangu模型路径"
    )
    
    # 判断是否默认启用外部API
    default_use_external = settings.external_api.enabled or bool(settings.external_api.api_key)
    
    parser.add_argument(
        "--use-external-api",
        action="store_true",
        default=default_use_external,
        help="使用外部API（如OpenAI）"
    )
    
    parser.add_argument(
        "--no-external-api",
        action="store_true",
        default=False,
        help="禁用外部API（即使配置了API密钥）"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=settings.external_api.api_key or os.getenv("EXTERNAL_API_KEY", ""),
        help="外部API密钥"
    )
    
    parser.add_argument(
        "--api-base",
        type=str,
        default=settings.external_api.api_base or os.getenv("EXTERNAL_API_BASE", ""),
        help="外部API地址"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default=settings.external_api.model_name or os.getenv("EXTERNAL_MODEL_NAME", "deepseek-ai/DeepSeek-V3.2"),
        help="外部API模型名称"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Gradio服务端口"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Gradio服务地址"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="创建公网分享链接"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("基于OpenPangu大模型的问答智能体 - 中期演示")
    logger.info("=" * 60)
    
    # 判断是否使用外部API
    use_external = args.use_external_api and not args.no_external_api
    
    # 如果没有通过命令行指定API密钥，但环境变量中有，则使用环境变量的
    api_key = args.api_key or settings.external_api.api_key
    api_base = args.api_base or settings.external_api.api_base
    model_name = args.model_name or settings.external_api.model_name
    
    logger.info(f"API配置: use_external={use_external}, api_key={'已设置' if api_key else '未设置'}, api_base={api_base}")
    
    agent = PanguAgent(
        model_path=args.model_path,
        use_external_api=use_external,
        external_api_key=api_key,
        external_api_base=api_base,
        external_model_name=model_name
    )
    
    logger.info("正在初始化智能体...")
    if not agent.initialize():
        logger.error("智能体初始化失败，退出")
        sys.exit(1)
    
    logger.info("正在启动Web界面...")
    interface = GradioInterface(agent)
    
    print("\n" + "=" * 60)
    print("🚀 服务启动成功！")
    print(f"📍 本地访问地址: http://localhost:{args.port}")
    if args.share:
        print("🌐 公网分享链接将在启动后显示")
    print("=" * 60 + "\n")
    
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
