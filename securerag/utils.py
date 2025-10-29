import logging
import sys
from typing import List


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
            pass  # logging 本身不需要 flush，这里留空

    sys.stdout = PrintToLogger(logger)


metric_map = {}


def add_metric(name, value):
    metric_map.setdefault(name, []).append(value)
    return value


def get_metric(name) -> List[int]:
    return metric_map[name]


def clear_metric(name) -> List[int]:
    return metric_map.pop(name, [])
