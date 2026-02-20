from securerag.config import Config
from securerag.scheduler.dispatcher import Dispatcher
from securerag.scheduler.requests import LocalRequestSource

config = Config()
config.load_size = 64
path = "data/open_domain_data/NQ/dev_with_scores.json"
local_request = LocalRequestSource()
local_request.registry_source(path, config)

dispatcher = Dispatcher(config)
dispatcher.registry_request_source(local_request)
dispatcher.endpoint_loop()
