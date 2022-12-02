import os
import random
import yaml

import torch
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def load_config(self, config_path, logger):
        with open(config_path, "r") as stream:
            cfg_dict = yaml.safe_load(stream)

        self.update(cfg_dict)

        logger.info(f"config arguments: {self}")

        # set random seed
        set_seed(self.random_state)
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.error("config conflicts: no gpu available, use cpu for inference.")
            self.device = -1
