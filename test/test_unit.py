import cProfile
import time
from test.test_base import TestConfigLoggerBase

import grpc
import torch
from google.protobuf import empty_pb2

from securerag.rpc import messages_pb2_grpc
from securerag.scheduler.dependency import FusionAggregate
from securerag.scheduler.dispatcher import Dispatcher
from securerag.scheduler.requests import LocalRequestSource
from securerag.scheduler.tasks import BatchEncoderTask, EncoderDecoderSerivce


class TestUnit(TestConfigLoggerBase):
    def test_local_request_source(self):
        path = "data/open_domain_data/NQ/dev_with_scores.json"
        local_request = LocalRequestSource()
        local_request.registry_source(path, self.config)

        for _ in range(3):
            time.sleep(1)
            reqs = local_request.arrive_requests()
            print("arrived request as follow:")
            for req in reqs:
                print(req.arrive_time)

    def test_tee_rpc_service(self):
        service = EncoderDecoderSerivce(self.config, "TEE")
        service.start_service()

    def test_gpu_rpc_service(self):
        service = EncoderDecoderSerivce(self.config, "GPU")
        service.start_service()

    def test_native_rpc_service(self):
        service = EncoderDecoderSerivce(self.config, "OFFLAODING")
        service.start_service()

    def test_native_rpc_client(self):
        port = 8082
        with grpc.insecure_channel(f"localhost:{port}") as channel:
            stub = messages_pb2_grpc.MetricServiceStub(channel)
            stub.ClearMetric(empty_pb2.Empty())

        self.config.load_size = 64
        path = "data/open_domain_data/NQ/dev_with_scores.json"
        local_request = LocalRequestSource()
        local_request.registry_source(path, self.config)

        dispatcher = Dispatcher(self.config)
        dispatcher.registry_request_source(local_request)
        dispatcher.native_endpoint_loop()

    def test_dispatcher_rpc_client(self):
        for port in [8080, 8081]:
            with grpc.insecure_channel(f"localhost:{port}") as channel:
                stub = messages_pb2_grpc.MetricServiceStub(channel)
                stub.ClearMetric(empty_pb2.Empty())

        self.config.load_size = 64
        path = "data/open_domain_data/NQ/dev_with_scores.json"
        local_request = LocalRequestSource()
        local_request.registry_source(path, self.config)

        dispatcher = Dispatcher(self.config)
        dispatcher.registry_request_source(local_request)
        dispatcher.endpoint_loop()

    def test_adaptive_passage_selection(self):
        public_scores = torch.Tensor([60]) / 100
        private_scores = torch.Tensor([80, 90, 85]) / 100
        ret = FusionAggregate.adaptive_passage_selection(
            public_scores, private_scores, 1
        )
        print(ret)
