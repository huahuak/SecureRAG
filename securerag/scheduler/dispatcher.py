import threading
from concurrent.futures import thread

import grpc

from securerag.rpc import messages_pb2_grpc
from securerag.scheduler.requests import RequestSource
from securerag.scheduler.tasks import (
    BatchDecoderTask,
    BatchEncoderTask,
    Task,
    TaskQueue,
)


class Dispatcher:
    def __init__(self, config):
        self.load_size = config.load_size
        self.finished_size = 0

        self.request_source = None

        self.tee_encoder_task_queue = TaskQueue()
        self.gpu_encoder_task_queue = TaskQueue()
        self.tee_decoder_task_queue = TaskQueue()
        self.gpu_decoder_task_queue = TaskQueue()

        self.running = True
        self.queue_lock = threading.Lock()

        self.tee_batch_size = 1
        self.gpu_batch_size = 32

    def registry_request_source(self, source: RequestSource):
        self.request_source = source

    def endpoint_loop(self, cfg):
        tee_loop_threead = threading.Thread(target=self.tee_loop, args=(cfg,))
        gpu_loop_threead = threading.Thread(target=self.gpu_loop, args=(cfg,))
        tee_loop_threead.start()
        gpu_loop_threead.start()
        while self.finished_size < self.load_size:
            if self.request_source is None:
                print("self.request_source must be registred")
                return
            # disaggregate request
            reqs = self.request_source.arrive_requests()
            self.queue_lock.acquire()
            for req in reqs:
                pub, pri = Task.create_task_from_request(req)
                self.tee_encoder_task_queue.append(pub)
                self.gpu_encoder_task_queue.append(pri)
            self.queue_lock.release()

    def tee_loop(self, cfg):
        port = cfg.encoder_service_port
        channel = grpc.insecure_channel(f"localhost:{port}")
        stub = messages_pb2_grpc.EncoderServiceStub(channel)
        while self.running:
            self.queue_lock.acquire()
            # disaggregate iteration
            encoder_task_waiting_time = self.tee_encoder_task_queue.total_waiting_time()
            decoder_task_waiting_time = self.tee_decoder_task_queue.total_waiting_time()

            if encoder_task_waiting_time >= decoder_task_waiting_time:
                # schedule request priority
                tasks = self.tee_encoder_task_queue.pop_shortest_tasks(self.tee_batch_size)
                batch = BatchEncoderTask().add_tasks(tasks)
                batch.rpc_execute(stub)
            else:
                pass
            self.queue_lock.release()

    def gpu_loop(self, cfg):
        while self.running:
            self.queue_lock.acquire()
            self.queue_lock.release()
