import cProfile
import os
import random
import threading
import time
from securerag.config import Config
from securerag.data import Profiler
from securerag.models import FiDT5
from test.test_base import TestConfigLoggerBase
from transformers.file_utils import ModelOutput

import grpc
import torch
from google.protobuf import empty_pb2

from securerag.rpc import messages_pb2_grpc
from securerag.scheduler.dependency import FusionAggregate
from securerag.scheduler.dispatcher import Dispatcher, WeakTEE
from securerag.scheduler.requests import (
    FixedTestRequestSource,
    LocalRequestSource,
    RpcRequestSource,
)
from securerag.scheduler.tasks import (
    BatchEncoderTask,
    EncoderDecoderSerivce,
    LocalEncoderDecoderService,
)
from securerag.utils import ProcessManager, clear_metric, delete_metric, dump_metric


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

    def test_gpu_rpc_service(self):
        service = EncoderDecoderSerivce(self.config, "GPU")
        service.start_service(worker_num=2)

    def test_offloading_rpc_client(self):
        ProcessManager.registry_interrupt("offloading")
        service = LocalEncoderDecoderService(self.config, "TEE")

        path = "data/open_domain_data/NQ/dev_with_scores.json"
        local_request = LocalRequestSource()
        local_request.registry_source(path, self.config)

        dispatcher = Dispatcher(self.config)
        dispatcher.registry_request_source(local_request)
        dispatcher.endpoint_loop_baseline(
            service, enable_offloading=True, enable_adaptive_fusion=False
        )

    def test_adpative_fusion_rpc_client(self):
        ProcessManager.registry_interrupt("adpative")
        service = LocalEncoderDecoderService(self.config, "TEE")

        path = "data/open_domain_data/NQ/dev_with_scores.json"
        local_request = LocalRequestSource()
        local_request.registry_source(path, self.config)

        dispatcher = Dispatcher(self.config)
        dispatcher.registry_request_source(local_request)
        dispatcher.endpoint_loop_baseline(
            service, enable_offloading=True, enable_adaptive_fusion=True
        )

    def test_native_rpc_client(self):
        ProcessManager.registry_interrupt("native_k10_s100")
        service = LocalEncoderDecoderService(self.config, "TEE")

        path = "data/open_domain_data/NQ/dev_with_scores.json"
        local_request = FixedTestRequestSource()
        local_request.registry_source(path, self.config)
        local_request.request_per_second = self.config.load_size
        local_request.set_private_ratio(self.config.n_context, 1)

        dispatcher = Dispatcher(self.config)
        dispatcher.registry_request_source(local_request)
        dispatcher.endpoint_loop_baseline(
            service, enable_offloading=False, enable_adaptive_fusion=False
        )

    def test_sched_rpc_client(self):
        ProcessManager.registry_interrupt()
        service = LocalEncoderDecoderService(self.config, "TEE")

        path = "data/open_domain_data/NQ/dev_with_scores.json"
        local_request = LocalRequestSource()
        local_request.registry_source(path, self.config)
        local_request.request_per_second = float(os.getenv("RPS", 1))

        dispatcher = Dispatcher(self.config)
        dispatcher.registry_request_source(local_request)
        dispatcher.endpoint_loop(service)

    def test_efficency(self):
        dispatcher = Dispatcher(self.config)
        for is_adaptive in [False, True]:
            for d in [0.1, 0.3, 0.5, 0.7, 0.9]:
                service = LocalEncoderDecoderService(self.config, "TEE")
                path = "data/open_domain_data/NQ/dev_with_scores.json"
                local_request = FixedTestRequestSource()
                local_request.registry_source(path, self.config)
                local_request.request_per_second = self.config.load_size
                local_request.set_private_ratio(self.config.n_context, d)

                dispatcher.registry_request_source(local_request)
                dispatcher.endpoint_loop_baseline(
                    service, enable_offloading=True, enable_adaptive_fusion=is_adaptive
                )
                clear_metric("finished_request_tee")
            dump_metric(f"tmp/is_adaptive_{is_adaptive}_thre0.9_k10_s100_e.json")

    def test_multi_rpc_client(self):
        service = LocalEncoderDecoderService(self.config, "TEE")
        weaktee = WeakTEE()

        path = "data/open_domain_data/NQ/dev_with_scores.json"
        local_request = LocalRequestSource()
        local_request.registry_source(path, self.config)
        local_request.request_per_second = float(os.getenv("RPS", 1))

        ProcessManager.registry_interrupt(
            # f"strong_tee12_8_request{local_request.request_per_second}"
        )

        dispatcher = Dispatcher(self.config)
        dispatcher.registry_request_source(local_request)
        dispatcher.registry_tee_rpc_instance(weaktee)
        dispatcher.endpoint_loop(service)

    def test_tee_instance(self):
        ProcessManager.registry_interrupt("weak")
        service = LocalEncoderDecoderService(self.config, "TEE")
        rpc_request = RpcRequestSource(self.config.tee_service_port)
        threading.Thread(target=rpc_request.start_service, daemon=True).start()
        dispatcher = Dispatcher(self.config)
        dispatcher.registry_request_source(rpc_request)
        dispatcher.endpoint_loop(service)

    def test_decoder_with_k_contexts(self):
        cfg = Config()
        model_path = cfg.generator_model_path
        model: FiDT5 = FiDT5.from_pretrained(model_path)
        batch_size = 10
        device = "cpu"
        encoder_outputs = ModelOutput()
        random.seed(2026)
        for k in range(10):
            contexts = torch.rand(
                (
                    batch_size,
                    200 * k,
                    768,
                )
            )
            encoder_outputs["last_hidden_state"] = contexts.to(device)
            masks = torch.full(
                (
                    batch_size,
                    200 * k,
                ),
                True,
            )
            with Profiler(f"DECODER"):
                for _ in range(10):
                    ans = model.generate_without_encoder(
                        input_ids=torch.empty(batch_size, 1).to(
                            device
                        ),  # useless input_ids
                        encoder_outputs=encoder_outputs,
                        attention_mask=masks.to(device),
                        max_length=2,
                    )
            dump_metric("tmp/decoder_with_k_contexts.json")
