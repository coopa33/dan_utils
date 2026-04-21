import pandas as pd
import numpy as np
import torch

from dan_utils import manual_multi_apply
from dan_utils import exp_dispatch, sum_dispatch
def perplexity(log_likelihoods, N: int) -> float:
    """Calculate perplexity from log-likelihood."""
    return exp_dispatch(sum_dispatch(1 / log_likelihoods)) ** (1 / N)

    

if __name__ == "__main__":
    # Example usage
    log_likelihoods = np.array([-0.5, -1.0, -1.5])  # Example log-likelihoods
    N = len(log_likelihoods)  # Number of tokens
    print(perplexity(log_likelihoods, N))
