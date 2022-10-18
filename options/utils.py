
import yaml
import os
import collections

from utils import logger
# from utils.download_utils import get_local_path

try:
    collections_abc = collections.abc  # 定义抽象基类
except AttributeError:
    collections_abc = collections

DEFAULT_CONFIG_DIR = "config"

def flatten_yaml_as_dict(cfg, parent_key="", sep = "."):
    items = []
    for k, v in cfg.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections_abc.MutableMapping):
            items.extend(flatten_yaml_as_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_config_file(opts):
    config_file_name = getattr(opts, "common.config_file", None)
    if config_file_name is None:
        return opts

    if not os.path.isfile(config_file_name):
        logger.error(
            "Configuration file does not exists at {}".format(config_file_name)
        )

    with open(config_file_name, "r") as yaml_file:
        try:
            cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

            flat_cfg = flatten_yaml_as_dict(cfg)
            for k, v in flat_cfg.items():
                if hasattr(opts, k):
                    setattr(opts, k, v)
        except yaml.YAMLError as exc:
            logger.error(
                "Error while loading config file: {}. Error message:{}".format(config_file_name, str(exc))
            )

    # override arguments
    override_args = getattr(opts, "override_args", None)
    if override_args is not None:
        for override_k, override_v in override_args.items():
            if hasattr(opts, override_k):
                setattr(opts, override_k, override_v)

    return opts
