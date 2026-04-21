import time
import functools
import numpy as np
import torch
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


def check_nan(func):
    """
    Silently checks whether function output contains NaN values. If so, raise error.
    Only works for functions that output either tensor or array"""

    @functools.wraps(func)
    def wrapper_check_nan(*args, **kwargs):

        result = func(*args, **kwargs)
        has_nan = False

        # pytorch and numpy logic for NaN
        # only torch tensors have method isnan()
        if hasattr(result, 'isnan'):
            # isnan() returns a boolean tensor, and any() checks for at least one NaN
            if result.isnan().any():
                has_nan = True

        elif isinstance(result, (np.ndarray, float)):
            if np.any(np.isnan(result)):
                has_nan = True
        
        if has_nan:
            raise ValueError(f"NaN detected in {func.__name__}")
        return result
    
    return wrapper_check_nan

    


    

if __name__=="__main__":
    @check_nan
    def produce_torch_nan():
        return torch.tensor([1.0, 3.0])
    
    @check_nan
    def produce_numpy_nan():
        return np.array([1.0, np.nan])
    
    produce_torch_nan()
    produce_numpy_nan()
    

    


