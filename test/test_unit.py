
import time
from test.test_base import TestConfigLoggerBase

from securerag.scheduler.dispatcher import Dispatcher
from securerag.scheduler.requests import LocalRequestSource
from securerag.scheduler.tasks import BatchEncoderTask


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
            
        
    def test_encoder_rpc_service(self):
        service = BatchEncoderTask()
        service.start_service(self.config)
    
    def test_encoder_rpc_client(self):
        path = "data/open_domain_data/NQ/dev_with_scores.json"
        local_request = LocalRequestSource()
        local_request.registry_source(path, self.config)

        dispatcher = Dispatcher(self.config)
        dispatcher.registry_request_source(local_request)
        dispatcher.endpoint_loop(self.config)
        
