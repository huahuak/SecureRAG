import unittest

from securerag.config import Config
from securerag.utils import init_logger, redirectPrintToLogger


class TestConfigLoggerBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config()
        logger = init_logger(filename=cls.config.log_path)
        redirectPrintToLogger()
