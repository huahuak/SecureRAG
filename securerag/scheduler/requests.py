import random
import time
from logging import info
from typing import List

from securerag import data


class Request:
    def __init__(self):
        self.arrive_time = None
        self.finish_time = None
        self.first_token_time = None
        self.token_between_time = []
        self.input_data = None
        self.private_passage_size = 0
        self.private_ans = None
        self.public_ans = None
        self.final_ans = None


class RequestSource:
    def arrive_requests(self) -> List[Request]:
        return []


class LocalRequestSource(RequestSource):
    def __init__(self):
        random.seed(2026)

        self.dataset = None
        self.curr = 0
        self.lasttime = time.time()
        self.request_per_second = 512

    def registry_source(self, path, config):
        datas = data.load(path=path, size=config.load_size)
        self.dataset = data.Dataset(data=datas, n_context=config.n_context)

    def arrive_requests(self) -> List[Request]:
        now = time.time()
        interval = now - self.lasttime

        request_size = int(interval * self.request_per_second)
        if request_size > 0:
            self.lasttime = time.time()
        data_size = len(self.dataset)

        # info(f"now: {now}, interval: {interval}, request_size: {request_size}.")

        output = []
        while self.curr < data_size and request_size > 0:
            req = Request()
            req.arrive_time = now
            req.input_data = self.dataset[self.curr]
            n_passages = len(req.input_data["passages"])
            req.private_passage_size = int(
                self.get_private_passage_ratio() * n_passages
            )
            output.append(req)
            request_size -= 1
            self.curr += 1

        return output

    def get_private_passage_ratio(self):
        return random.random()
