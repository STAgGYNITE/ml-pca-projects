"""
Custom scaler utilities wrapping sklearn for pipeline consistency.
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
from typing import Union


def scale_features(
    X: Union[np.ndarray, pd.DataFrame],
    method: str = "standard"
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Scale features using standard or minmax scaling.

    Parameters
    ----------
    X : array-like
    method : str — 'standard' or 'minmax'
    """
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler()
    }
    if method not in scalers:
        raise ValueError(f"method must be one of {list(scalers.keys())}")

    scaler = scalers[method]
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return scaler.fit_transform(X)
