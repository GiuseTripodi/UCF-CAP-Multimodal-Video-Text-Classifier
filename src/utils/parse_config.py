import inspect
import logging
import logging.config
import os
import time
from datetime import datetime
from functools import reduce
from operator import getitem
from pathlib import Path
import json

class ConfigParser:
    def __init__(self, args, timestamp=True, test=False):
        # load json config file
        if args.config is None:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
        with open(args.config, 'r') as file:
            self.config = json.load(file)

        # set save_dir where trained model and log will be saved.
        save_dir = args.save_dir

        if args.name is None:
            self._exper_name = self.config['name']
        else:
            self._exper_name = args.name

        self._save_dir = os.path.join(save_dir, 'models')
        self._log_dir = os.path.join(save_dir, 'log')
        self.data_dir = os.path.join(save_dir, 'UcfCap')

        # get model parameters
        self.model_parameters = self.config['trainer']

    def __getitem__(self, name):
        return self.config[name]

    def get_logger(self, name, ):
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "simple": {"format": "%(message)s"},
                "datetime": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "DEBUG",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout",
                },
                "info_file_handler": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "datetime",
                    "filename": "info.log",
                    "maxBytes": 10485760,
                    "backupCount": 20,
                    "encoding": "utf8",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["console", "info_file_handler"],
            },
        }

        logging.config.dictConfig(log_config)
        logger = logging.getLogger(name)
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def batch_size(self):
        return self.config['data_loader']['batch_size']

    @property
    def shuffle(self):
        return self.config['data_loader']['shuffle']


    @property
    def train_path(self):
        train_path = os.path.join(self.data_dir, 'train_dataset.csv')
        val_path = os.path.join(self.data_dir, 'val_dataset.csv')
        return train_path, val_path

    @property
    def test_path(self):
        test_path = os.path.join(self.data_dir, 'test_dataset.csv')
        return test_path

    @property
    def train_data_path(self):
        return self._log_dir

    @property
    def model_parameters(self):
        return self._model_parameters

    @property
    def img_size(self):
        return self.config['trainer']['img_size']
    @property
    def in_chans(self):
        return self.config['trainer']['in_chans']
    @property
    def num_frames(self):
        return self.config['trainer']['num_frames']
    @property
    def num_classes(self):
        return self.config['trainer']['num_classes']
    @property
    def depth(self):
        return self.config['trainer']['depth']
    @property
    def num_heads(self):
        return self.config['trainer']['num_heads']
    @property
    def num_epochs(self):
        return self.config['trainer']['epochs']
    @property
    def max_seq_len(self):
        return self.config['trainer']['max_seq_len']

    @property
    def learning_rate(self):
        return self.config['optimizer']['args']['lr']

    @property
    def exper_name(self):
        return self._exper_name

    @property
    def model_name(self):
        return self.config['trainer']['model_name']
    @property
    def modality(self):
        return self.config['trainer']['modality']

    @model_parameters.setter
    def model_parameters(self, value):
        self._model_parameters = value

    @config.setter
    def config(self, value):
        self._config = value
