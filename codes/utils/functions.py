import argparse
from omegaconf import OmegaConf


def setup_parser(config_path=None):
    parser = argparse.ArgumentParser(
        description='多模态实体链接双塔模型训练',
        add_help=True
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/wikimel_dual_tower.yaml',
        help='配置文件路径'
    )

    _args = parser.parse_args()
    config_file = config_path or _args.config

    args = OmegaConf.load(config_file)

    return args
