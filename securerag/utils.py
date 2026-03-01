import json
import logging
import os
import signal
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
            # self.logger.log(self.level, f"{message}", stacklevel=2)

        def flush(self):
            pass

    sys.stdout = PrintToLogger(logger)


class ProcessManager:

    def registry_interrupt(process_name=None):
        def handler(sig, frame):
            dump_metric(f"tmp/{process_name}.json")
            sys.exit(0)

        if process_name is None:
            process_name = os.getenv("PROCESS_NAME")

        add_metric("process_name", process_name)
        signal.signal(signal.SIGINT, handler)


metric_map = {}


def add_metric(name, value):
    metric_map.setdefault(name, []).append(value)
    return value


def get_metric(name) -> List[int]:
    return metric_map.get(name, [])


def clear_metric(name) -> List[int]:
    return metric_map.pop(name, [])


def delete_metric():
    metric_map.clear()


def dump_metric(filepath):
    with open(filepath, "w", encoding="utf-8") as file:
        data = {
            "raw": metric_map,
            "statics": {
                k + "(statics)": [np.mean(v), np.min(v), np.max(v), np.sum(v)]
                for k, v in sorted(metric_map.items())
                if len(v) > 0 and not isinstance(v[0], str)
            },
        }
        json.dump(data, file, indent=2, default=str)


def show_metric():
    def map_print(data):
        formatted = json.dumps(
            data, indent=4, ensure_ascii=False, default=lambda o: float(o)
        )
        print("\n" + formatted)

    # map_print(metric_map)
    map_print(
        {
            k + "(statics)": [np.mean(v), np.min(v), np.max(v), np.sum(v)]
            for k, v in sorted(metric_map.items())
            if len(v) > 0 and not isinstance(v[0], str)
        }
    )
