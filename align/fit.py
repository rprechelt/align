"""
Various fast fit functions for NumPy arrays.
"""
import numba
import numpy as np

__all__ = ["fit_poly", "fit_gauss"]


@numba.njit
def fit_poly(x: np.ndarray) -> float:
    """
    Peform a 3-point quadratic interpolation to find
    the maximum of `x`.

    Parameters
    ----------
    x: np.ndarray
        A length-3 array centered around the maximum

    Returns
    -------
    float:
        The location of the maximum in samples w.r.t to the start of `x`
    """
    return (x[2] - x[0]) / (2 * (2 * x[1] - x[0] - x[2]))


@numba.njit
def fit_gauss(x: np.ndarray) -> float:
    """
    Peform a 3-point Gaussian interpolation to find
    the maximum of `x`.

    Parameters
    ----------
    x: np.ndarray
        A length-3 array centered around the maximum

    Returns
    -------
    float:
        The location of the maximum in samples w.r.t to the start of `x`
    """
    return (np.log(x[2]) - np.log(x[0])) / (
        4 * np.log(x[1]) - 2 * np.log(x[0]) - 2 * np.log(x[2])
    )
