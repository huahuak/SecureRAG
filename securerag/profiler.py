from torch.profiler import record_function
from functools import wraps
import time

from securerag.utils import add_metric


class Profiler:
    def __init__(self, name, print_time=False):
        self.name = name
        self.ctx = None
        self.print_time = print_time

    def __enter__(self):
        self.ctx = record_function(self.name)
        self.start = time.time()
        self.ctx.__enter__() 
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        self.ctx.__exit__(exc_type, exc_val, exc_tb)
        add_metric(self.name, end - self.start)
        if self.print_time:
            print(f"[Profiler] {self.name}: {end - self.start:.4f} seconds")
        return False 

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self: 
                return func(*args, **kwargs)
        return wrapper

