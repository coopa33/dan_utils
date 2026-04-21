import pandas as pd
import numpy as np
import torch

from typing import Dict

def manual_multi_apply(data: pd.DataFrame | np.ndarray | torch.Tensor, funcs: Dict[type, callable]):
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
