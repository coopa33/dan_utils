# Global imports
import pandas as pd
import numpy as np
import torch
from typing import Dict

# Local imports
from dan_utils.diagnostics import console as colors
from dan_utils.diagnostics.console import print_begin, print_end 
from dan_utils.data.impute import na_mask
from dan_utils.data.apply import manual_multi_apply


def multi_apply(data: pd.DataFrame | np.ndarray | torch.Tensor, funcs: Dict[type, callable]):
    """ Applies the appropriate function from the funcs dictionary based on the type of the input data.

    Args:
        data (pd.DataFrame | np.ndarray | torch.Tensor): The input data for which the appropriate function needs to be applied.
        funcs (Dict[type, callable], optional): A dictionary mapping types to their corresponding functions. Defaults to None.

    Raises:
        TypeError: If the input data type is not supported or if funcs is not a dictionary.
        KeyError: If no function is provided for the detected data type in the funcs dictionary.

    Returns:
        None | pd.DataFrame | np.ndarray | torch.Tensor: The result of applying the appropriate function to the input data.
    """
    types = (pd.DataFrame, np.ndarray, torch.Tensor)
    matches = {t: isinstance(data, t) for t in types}

    if not isinstance(funcs, dict):
        raise TypeError("funcs must be a dictionary mapping types to their corresponding functions.")
    if sum(matches.values()) == 0:
        raise TypeError("Input must be a pandas DataFrame, numpy array, or torch tensor.")
    
    
    if matches[pd.DataFrame]:
        if pd.DataFrame not in funcs:
            raise KeyError("No function provided for pandas DataFrame in funcs dictionary.")
        return funcs[pd.DataFrame](data)
    
    elif matches[np.ndarray]:
        if np.ndarray not in funcs:
            raise KeyError("No function provided for numpy array in funcs dictionary.")
        return funcs[np.ndarray](data)
    
    elif matches[torch.Tensor]:
        if torch.Tensor not in funcs:
            raise KeyError("No function provided for torch tensor in funcs dictionary.")
        return funcs[torch.Tensor](data)

def diagnose_data(
        data: pd.DataFrame | np.ndarray | torch.Tensor, 
        name: str, 
        get: bool = False
        ) -> None | pd.DataFrame | np.array | torch.Tensor:
    """
    Prints out basic diagnostics about the input data, including:
    - Data type (pandas DataFrame, numpy array, or torch tensor) | Value type (e.g., int, float, object)
    - Shape 
    - Number of missing values (NaNs) in the data
    - Percentage of missing values in the data
    """

    # Output header
    title_length = print_begin(f"Data Diagnostics")

    # Compute diagnostics based on data type
    if isinstance(data, pd.DataFrame):
        data_type = "pandas DataFrame"
        val_type = ", ".join([str(vt) for vt in data.dtypes.unique() if vt != 'object'])
        shape = data.shape
        num_missing = data.isna().sum().sum()
        total_elements = data.size

    elif isinstance(data, np.ndarray):
        data_type = "numpy array"
        val_type = data.dtype
        shape = data.shape
        num_missing = np.isnan(data).sum()
        total_elements = data.size

    elif isinstance(data, torch.Tensor):
        data_type = "torch tensor"
        val_type = data.dtype
        shape = tuple(data.shape)
        num_missing = torch.isnan(data).sum().item()
        total_elements = data.numel()
    else:
        raise TypeError("Input must be a pandas DataFrame, numpy array, or torch tensor.")
    
    percent_missing = (num_missing / total_elements) * 100 if total_elements > 0 else 0
    
    # Print general diagnostic information
    print(f"Diagnostics for {colors.red_str(name)}:") 
    print(f"{data_type} {colors.yellow_str('| ' + str(val_type)) if 'val_type' in locals() else ''}")
    print(f"Shape: {shape}")
    print(f"Number of Missing Values: {num_missing}")
    print(f"Percentage of Missing Values: {percent_missing:.2f}%")

    # Output footer
    print_end(title_length)
    # Newline
    print("\n")

    # Getters
    if get:
        return {
            "data_type": data_type,
            "shape": shape,
            "num_missing": num_missing,
            "percent_missing": percent_missing
        }

if __name__=="__main__":
    # Example usage
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [np.nan, 2, 3, 4],
        'C': ['a', 'b', 'c', 'd']
    })
    diagnose_data(df, name="df")

    arr = np.array([[1, 2], [3, 1]])
    diagnose_data(arr, name="arr")

    tensor = torch.tensor([[1.0, 2.0], [3.0, float('nan')]])
    diagnose_data(tensor, name="tensor")

    remove_na_dict = {
        pd.DataFrame: lambda df: df.dropna(axis=1),  # Drop rows with NaN values
        np.ndarray: lambda arr: arr[~np.isnan(arr)],
        torch.Tensor: lambda tensor: tensor[~torch.isnan(tensor)]
    }
    
    data_no_na = multi_apply(df, remove_na_dict)
    print( data_no_na)
