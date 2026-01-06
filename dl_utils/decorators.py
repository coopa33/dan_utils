import time
import functools
import pdb

def timer(func):
    """Print the runtime of the decorated function"""
    # define the timer to wrap around original function. @functools.wraps ensures metadate of original function is maintained
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()

        # exectute original function with parameters and return value
        value = func(*args, **kwargs)

        end_time = time.perf_counter()
        run_time = end_time - start_time

        print(f"Finished {func.__name__!r} in {run_time:.4f} sec")
        return value
    
    return wrapper_timer
    


    

if __name__=="__main__":
    pass

    


