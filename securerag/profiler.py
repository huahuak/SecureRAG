from torch.profiler import record_function
from functools import wraps


def profiler(name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with record_function(name):
                return func(*args, **kwargs)
        return wrapper

    return decorator
