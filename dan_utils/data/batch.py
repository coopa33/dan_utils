import numpy as np
import torch

def create_batches(X, y, batch_size=32, shuffle=True):
    """Batch generator for numpy arrays and pytorch tensors. Returns batch VIEWS of the original data"""

    sample_size = len(X)
    indices = np.arange(sample_size)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for i in range(0, sample_size, batch_size):
        batch_idx = indices[i : i+batch_size]
        yield X[batch_idx], y[batch_idx]
    