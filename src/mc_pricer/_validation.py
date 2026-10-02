"""Small shared scalar validation helpers."""

from math import isfinite
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


def finite_real(name: str, value: float) -> float:
    """Return a finite real scalar as float; reject booleans and arrays."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def positive_integer(name: str, value: int) -> int:
    """Validate counts before allocating arrays or advancing random streams."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def finite_vector(name: str, values: ArrayLike, *, min_size: int = 1) -> NDArray[np.float64]:
    """Validate a 1-D real finite vector and convert it to float64.

    May return a view of the input. Callers must not mutate the returned array
    unless they explicitly intend to mutate the original input too.
    """
    array = np.asarray(values)
    if array.ndim != 1 or array.size < min_size:
        raise ValueError(f"{name} must have shape (n,) with at least {min_size} elements")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    with np.errstate(over="raise", invalid="raise"):
        array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def finite_matrix(name: str, values: ArrayLike) -> NDArray[np.float64]:
    """Validate real finite shape (n_rows, n_columns), both dimensions >= 1.

    The result may alias the input, including noncontiguous/read-only arrays.
    Consumers must not mutate it without explicitly owning that storage.
    """
    array = np.asarray(values)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a nonempty two-dimensional array")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    with np.errstate(over="raise", invalid="raise"):
        array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array
