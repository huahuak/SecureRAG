import json
import logging
import sys
from typing import List
import numpy as np


logger = logging.getLogger(__name__)


def init_logger(filename=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    return logger


def redirectPrintToLogger():
    class PrintToLogger:
        def __init__(self, logger, level=logging.INFO):
            self.logger: logging.Logger = logger
            self.level = level
            self._buffer = ""

        def write(self, message):
            message = message.strip()
            if message:  # 过滤空行
                self.logger.log(self.level, f"{message}", stacklevel=2)

        def flush(self):
            pass

    sys.stdout = PrintToLogger(logger)


metric_map = {}


def add_metric(name, value):
    metric_map.setdefault(name, []).append(value)
    return value


def get_metric(name) -> List[int]:
    return metric_map.get(name, [])


def clear_metric(name) -> List[int]:
    return metric_map.pop(name, [])


def show_metric():
    def map_print(data):
        formatted = json.dumps(data, indent=4, ensure_ascii=False, default=lambda o: float(o))
        print("\n"+formatted)
    map_print(metric_map)
    map_print({k: [np.mean(v), np.min(v), np.max(v), np.sum(v)] for k, v in metric_map.items()})

