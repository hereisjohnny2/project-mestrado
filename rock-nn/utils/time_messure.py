import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        original_return_val = func(*args, **kwargs)
        end = time.time()
        time_elapsed = end - start
        print("time elapsed in ", func.__name__, ": ", time_elapsed, sep='')
        return original_return_val, time_elapsed

    return wrapper