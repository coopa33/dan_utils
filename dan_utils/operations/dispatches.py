import torch
import numpy as np
import pandas as pd

from functools import singledispatch


@singledispatch
def exp_dispatch(value):
    raise NotImplementedError(f"Unsupported type: {type(value)}")

@exp_dispatch.register
def _(value: float|np.ndarray|pd.DataFrame):
    return np.exp(value)

@exp_dispatch.register
def _(value: torch.Tensor):
    return torch.exp(value)


@singledispatch
def sum_dispatch(value, axis=None):
    raise NotImplementedError(f"Unsupported type: {type(value)}")

@sum_dispatch.register
def _(value: float|np.ndarray|pd.DataFrame, axis=None):
    return np.sum(value, axis=axis)

@sum_dispatch.register
def _(value: torch.Tensor, axis=None):
    if axis is not None:
        return torch.sum(value, dim=axis)
    return torch.sum(value)
