from torch.profiler import record_function
from functools import wraps
import time


def profiler(name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            with record_function(name):
                return func(*args, **kwargs)
            end = time.time()
            print(f"{name} elapsed: {end - start:.4f} seconds")
        return wrapper

    return decorator
