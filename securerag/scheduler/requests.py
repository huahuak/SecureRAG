import random
import time
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
        self.request_per_second = 1

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

        output = []
        while self.curr < data_size and request_size > 0:
            req = Request()
            req.input_data = self.dataset[self.curr]
            req = self.request_load_post_process(req)
            req.arrive_time = time.time()
            output.append(req)
            request_size -= 1
            self.curr += 1

        return output

    def request_load_post_process(self, req):  # -> Any:
        n_passages = len(req.input_data["passages"])
        siz = (
            min(random.randint(5, 10), n_passages)
            if random.random() < 0.8
            else min(random.randint(20, 50), n_passages)
        )
        req.private_passage_size = (
            random.randint(0, int(siz * 0.6))
            if random.random() < 0.9
            else random.randint(int(siz * 0.6), siz)
        )
        req.input_data["passages"] = req.input_data["passages"][:siz]
        req.input_data["scores"] = req.input_data["scores"][:siz]

        return req
