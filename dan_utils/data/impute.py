import pandas as pd
import numpy as np
import torch
from functools import singledispatch
from typing import Dict
"""
remove_na: 
- If this becomes relevant for listwise deletion in ML/DL contexts, it would
  be nice to have a variant or option which removes only rows which have NA
  in the target variable.

"""
@singledispatch
def remove_na(data, axis: int = 1):
    """ Removes NA values from the input data. The specific method of removal depends on the type of the input data (pandas DataFrame, numpy array, or torch tensor).

    Args:
        data (pd.DataFrame | np.ndarray | torch.Tensor): The input data from which missing values need to be removed.
        axis (int, optional): The axis along which to remove NA values. Defaults to 1.

    Raises:
        NotImplementedError: If the input data type is not supported (i.e., not a pandas DataFrame, numpy array, or torch tensor).

    Returns:
        pd.DataFrame | np.ndarray | torch.Tensor: The input data with missing values removed, in the same format as the input.
    """
    raise NotImplementedError(f"remove_na not implemented for type {type(data)}")

@remove_na.register(pd.DataFrame)
def _(data: pd.DataFrame, axis: int = 1):
    return data.dropna(axis=axis)    

@remove_na.register(np.ndarray)
def _(data: np.ndarray, axis: int = 1):
    if data.ndim == 1:
        return data[~np.isnan(data)]
    
    if axis == 0:
        kept_rows = ~np.isnan(data).any(axis=1)
        return data[kept_rows, :]
    elif axis == 1:
        kept_cols = ~np.isnan(data).any(axis=0)
        return data[:, kept_cols]
    else:
        raise ValueError("Axis must be 0 (rows) or 1 (columns) for numpy arrays.")
    
@remove_na.register(torch.Tensor)
def _(data: torch.Tensor, axis: int = 1):
    return data[~torch.isnan(data)]


def na_mask(df: pd.DataFrame | np.array | torch.Tensor) -> pd.DataFrame | np.array | torch.Tensor:
    """
    Returns a boolean mask indicating the locations of NaN values in the input DataFrame or numpy array.
    For DataFrames, it returns a DataFrame of the same shape with True for NaN values and False otherwise.
    For numpy arrays, it returns a boolean array of the same shape with True for NaN values and False otherwise.
    """
    
    if isinstance(df, pd.DataFrame):
        return df.isna()
    elif isinstance(df, np.ndarray):
        return np.isnan(df)
    elif isinstance(df, torch.Tensor):
        return torch.isnan(df)
    else:
        raise TypeError("Input must be a pandas DataFrame, numpy array, or torch tensor.")



def impute_mean(df: pd.DataFrame | np.array | torch.Tensor) -> pd.DataFrame | np.array | torch.Tensor:
    """
    Imputes missing values in a DataFrame or numpy array using mean imputation.
    For DataFrames, it imputes each column separately. For numpy arrays, it imputes the entire array.
    """
    
    if isinstance(df, pd.DataFrame):
        return df.fillna(df.mean())
    elif isinstance(df, np.ndarray):
        # Compute the mean of each column, ignoring NaNs
        col_means = np.nanmean(df, axis=0)
        # Find indices where NaNs are present
        inds = np.where(np.isnan(df))
        # Replace NaNs with the corresponding column means
        df[inds] = np.take(col_means, inds[1])
        return df
    elif isinstance(df, torch.Tensor):
        # Compute the mean of each column, ignoring NaNs
        col_means = torch.nanmean(df, dim=0)
        # Find indices where NaNs are present
        inds = torch.where(torch.isnan(df))
        # Replace NaNs with the corresponding column means
        df[inds] = torch.take(col_means, inds[1])
        return df
    else:
        raise TypeError("Input must be a pandas DataFrame, numpy array, or torch tensor.")

def impute_median(df: pd.DataFrame | np.array | torch.Tensor) -> pd.DataFrame | np.array | torch.Tensor:
    """
    Imputes missing values in a DataFrame or numpy array using median imputation.
    For DataFrames, it imputes each column separately. For numpy arrays, it imputes the entire array.
    """
    
    if isinstance(df, pd.DataFrame):
        return df.fillna(df.median())
    elif isinstance(df, np.ndarray):
        # Compute the median of each column, ignoring NaNs
        col_medians = np.nanmedian(df, axis=0)
        # Find indices where NaNs are present
        inds = np.where(np.isnan(df))
        # Replace NaNs with the corresponding column medians
        df[inds] = np.take(col_medians, inds[1])
        return df
    elif isinstance(df, torch.Tensor):
        # Compute the median of each column, ignoring NaNs
        col_medians = torch.nanmedian(df, dim=0)
        # Find indices where NaNs are present
        inds = torch.where(torch.isnan(df))
        # Replace NaNs with the corresponding column medians
        df[inds] = torch.take(col_medians, inds[1])
        return df
    else:
        raise TypeError("Input must be a pandas DataFrame, numpy array, or torch tensor.")
