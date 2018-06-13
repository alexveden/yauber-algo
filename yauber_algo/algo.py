"""
Public accessor for algorithmic indicator codebase
"""
import os

from .errors import YaUberAlgoArgumentError, YaUberAlgoFutureReferenceError, YaUberAlgoInternalError, YaUberAlgoDtypeNotSupportedError
from ._algo.tstoolbox import _ref, _iif, _hhv, _llv, _highest_since, _lowest_since, _bars_since, \
                                  _cross_up, _cross_dn, _sum, _ma, _stdev, _sum_since, _zscore, _min, \
                                  _max, _abs, _value_when, _nz, _roc, _diff, _rsi, _rangehilo, _rangeclose, \
                                  _wma, _correlation, _truerange, _updn_ratio, _roc_log

from ._algo.featurewiz import _percent_rank, _percent_rank_category, cat_sort_unique, _apply_categorical, _apply_rolling, _quantile, \
                                   _categorize


import warnings
import numpy as np
import pandas as pd
from math import  isfinite
pd.options.mode.use_inf_as_na = True

IS_WARN_FUTREF = os.getenv('YAUBER_WARN_FUTREF', False)
IS_RAISE_FUTREF = os.getenv('YAUBER_RAISE_FUTREF', False)


def _check_series_args(**kwargs):
    arr_len = -1
    arg_type = None
    for arg_n, arg_val in kwargs.items():
        if not isinstance(arg_val, (pd.Series, np.ndarray)):
            raise YaUberAlgoArgumentError(f"Argument <{arg_n}> must be pd.Series or np.ndarray, got {type(arg_val)}")

        if arg_val.dtype not in (np.float, np.bool, np.int32, np.int64, np.float32):
            raise YaUberAlgoDtypeNotSupportedError(f'Argument <{arg_n}> must have dtype in (np.float, np.int32, np.int64, np.bool), '
                                                  f'got {arg_val.dtype}')
        if arr_len != -1:
            if arr_len != len(arg_val):
                raise YaUberAlgoArgumentError("Arguments have different lengths.")
        else:
            arr_len = len(arg_val)

        if arg_type is not None:
            if type(arg_val) != arg_type:
                raise YaUberAlgoArgumentError(f'Arguments must have the same type, got: {type(arg_val)} vs previous {arg_type}')
        else:
            arg_type = type(arg_val)


def _get_series_or_number(arr_or_number, return_like):
    try:
        # Check if arr is iterable
        iter(arr_or_number)
        return arr_or_number
    except TypeError:
        if isinstance(return_like, pd.Series):
            return pd.Series(arr_or_number, index=return_like.index)
        else:
            return np.full(len(return_like), arr_or_number)


def ref(arr, period):
    if period > 0:
        if IS_WARN_FUTREF:
            warnings.warn(f"Future reference detected in ref(), period: {period}")
        if IS_RAISE_FUTREF:
            raise YaUberAlgoFutureReferenceError(f"Future reference detected in ref(), period: {period}")

    # Do quick sanity checks of arguments
    _check_series_args(arr=arr)

    if isinstance(arr, pd.Series):
        return pd.Series(_ref(arr.values, period), index=arr.index)
    elif isinstance(arr, np.ndarray):
        return _ref(arr, period)


def iif(cond, if_true_arr, if_false_arr):
    """

    :param cond:
    :param if_true_arr: array or number
    :param if_false_arr: array or number
    :return:
    """
    try:
        _if_true_arr = _get_series_or_number(if_true_arr, cond)
        _if_false_arr = _get_series_or_number(if_false_arr, cond)

        # Do quick sanity checks of arguments
        _check_series_args(cond=cond, if_true_arr=_if_true_arr, if_false_arr=_if_false_arr)

        if isinstance(cond, pd.Series):
            return pd.Series(_iif(cond.values, _if_true_arr.values, _if_false_arr.values), index=cond.index)
        elif isinstance(cond, np.ndarray):
            return _iif(cond, _if_true_arr, _if_false_arr)

    except ValueError as exc:
        raise YaUberAlgoInternalError(str(exc))


def hhv(arr, period):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _check_series_args(arr=arr)

    if isinstance(arr, pd.Series):
        return pd.Series(_hhv(arr.values, period), index=arr.index)
    elif isinstance(arr, np.ndarray):
        return _hhv(arr, period)


def llv(arr, period):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _check_series_args(arr=arr)

    if isinstance(arr, pd.Series):
        return pd.Series(_llv(arr.values, period), index=arr.index)
    elif isinstance(arr, np.ndarray):
        return _llv(arr, period)


def highest_since(arr, cond):
    # Do quick sanity checks of arguments
    _check_series_args(cond=cond, arr=arr)

    try:
        if isinstance(arr, pd.Series):
            return pd.Series(_highest_since(arr.values, cond.values), index=arr.index)
        elif isinstance(arr, np.ndarray):
            return _highest_since(arr, cond)
    except ValueError as exc:
        raise YaUberAlgoInternalError(str(exc))


def lowest_since(arr, cond):
    # Do quick sanity checks of arguments
    _check_series_args(cond=cond, arr=arr)

    try:
        if isinstance(arr, pd.Series):
            return pd.Series(_lowest_since(arr.values, cond.values), index=arr.index)
        elif isinstance(arr, np.ndarray):
            return _lowest_since(arr, cond)
    except ValueError as exc:
        raise YaUberAlgoInternalError(str(exc))


def bars_since(cond):
    # Do quick sanity checks of arguments
    _check_series_args(cond=cond)
    try:
        if isinstance(cond, pd.Series):
            return pd.Series(_bars_since(cond.values), index=cond.index)
        elif isinstance(cond, np.ndarray):
            return _bars_since(cond)
    except ValueError as exc:
        raise YaUberAlgoInternalError(str(exc))


def cross_up(arr, arr_threshold):
    """
    Crosses up if arr[i-1] < arr_threshold[i-1] and arr[i] > arr_threshold[i]
    :param arr:
    :param arr_threshold: array or number
    :return:
    """
    _arr_threshold = _get_series_or_number(arr_threshold, arr)

    # Do quick sanity checks of arguments
    _check_series_args(arr_threshold=_arr_threshold, arr=arr)

    if isinstance(_arr_threshold, pd.Series):
        return pd.Series(_cross_up(arr.values, _arr_threshold.values), index=_arr_threshold.index)
    elif isinstance(_arr_threshold, np.ndarray):
        return _cross_up(arr, _arr_threshold)


def cross_dn(arr, arr_threshold):
    """
    Crosses down if arr[i-1] > arr_threshold[i-1] and arr[i] < arr_threshold[i]
    :param arr:
    :param arr_threshold: array or number
    :return:
    """
    _arr_threshold = _get_series_or_number(arr_threshold, arr)

    # Do quick sanity checks of arguments
    _check_series_args(arr_threshold=_arr_threshold, arr=arr)

    if isinstance(_arr_threshold, pd.Series):
        return pd.Series(_cross_dn(arr.values, _arr_threshold.values), index=_arr_threshold.index)
    elif isinstance(_arr_threshold, np.ndarray):
        return _cross_dn(arr, _arr_threshold)


def sum(arr, period):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _check_series_args(arr=arr)

    if isinstance(arr, pd.Series):
        return pd.Series(_sum(arr.values, period), index=arr.index)
    elif isinstance(arr, np.ndarray):
        return _sum(arr, period)


def ma(arr, period):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _check_series_args(arr=arr)

    if isinstance(arr, pd.Series):
        return pd.Series(_ma(arr.values, period), index=arr.index)
    elif isinstance(arr, np.ndarray):
        return _ma(arr, period)


def stdev(arr, period):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _check_series_args(arr=arr)

    if arr.dtype == np.bool:
        raise YaUberAlgoDtypeNotSupportedError("Boolean dtype is not supported")

    try:
        if isinstance(arr, pd.Series):
            return pd.Series(_stdev(arr.values, period), index=arr.index)
        elif isinstance(arr, np.ndarray):
            return _stdev(arr, period)
    except ValueError as exc:
            raise YaUberAlgoInternalError(str(exc))


def percent_rank(arr, period):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _check_series_args(arr=arr)

    try:
        if isinstance(arr, pd.Series):
            return pd.Series(_percent_rank(arr.values, period), index=arr.index)
        elif isinstance(arr, np.ndarray):
            return _percent_rank(arr, period)
    except ValueError as exc:
            raise YaUberAlgoInternalError(str(exc))


def percent_rank_category(arr, category, period):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _check_series_args(arr=arr, category=category)

    sorted_cat_arr = cat_sort_unique(category)

    if len(sorted_cat_arr) > 100:
        raise YaUberAlgoArgumentError("Too many categories, no more than 100 allowed")

    try:
        if isinstance(arr, pd.Series):
            return pd.Series(_percent_rank_category(arr.values, category.values, sorted_cat_arr, period), index=arr.index)
        elif isinstance(arr, np.ndarray):
            return _percent_rank_category(arr, category, sorted_cat_arr, period)
    except ValueError as exc:
            raise YaUberAlgoInternalError(str(exc))


def apply(arr, period, func, category=None, return_as_cat=None, exclude_nan=True):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _kwargs = {}
    _kwargs['arr'] = arr

    _return_as_cat = None

    if category is not None:
        _kwargs['category'] = category
    if return_as_cat is not None:
        _return_as_cat = _get_series_or_number(return_as_cat, arr)
        _kwargs['return_as_cat'] = _return_as_cat

    _check_series_args(**_kwargs)

    if category is not None:
        sorted_cat_arr = cat_sort_unique(category)

        if len(sorted_cat_arr) > 100:
            raise YaUberAlgoArgumentError("Too many categories, no more than 100 allowed")

        if isinstance(arr, pd.Series):
            return pd.Series(_apply_categorical(arr.values, period, func,
                                                category.values, sorted_cat_arr,
                                                None if _return_as_cat is None else _return_as_cat.values,
                                                exclude_nan), index=arr.index)
        elif isinstance(arr, np.ndarray):
            return _apply_categorical(arr, period, func,
                                      category, sorted_cat_arr, _return_as_cat, exclude_nan)

    else:
        if isinstance(arr, pd.Series):
            return pd.Series(_apply_rolling(arr.values, period, func, exclude_nan), index=arr.index)
        elif isinstance(arr, np.ndarray):
            return _apply_rolling(arr, period, func, exclude_nan)


def sum_since(arr, cond, first_is_zero=False):
    # Do quick sanity checks of arguments
    _check_series_args(arr=arr, cond=cond)

    try:
        if isinstance(cond, pd.Series):
            return pd.Series(_sum_since(arr.values, cond.values, first_is_zero), index=cond.index)
        elif isinstance(cond, np.ndarray):
            return _sum_since(arr, cond, first_is_zero)
    except ValueError as exc:
            raise YaUberAlgoInternalError(str(exc))


def zscore(arr, period):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _check_series_args(arr=arr)
    try:
        if isinstance(arr, pd.Series):
            return pd.Series(_zscore(arr.values, period), index=arr.index)
        elif isinstance(arr, np.ndarray):
            return _zscore(arr, period)
    except ValueError as exc:
            raise YaUberAlgoInternalError(str(exc))


def min(arr1, arr2):

    # Do quick sanity checks of arguments
    _check_series_args(arr1=arr1, arr2=arr2)

    if isinstance(arr1, pd.Series):
        return pd.Series(_min(arr1.values, arr2.values), index=arr1.index)
    elif isinstance(arr1, np.ndarray):
        return _min(arr1, arr2)


def max(arr1, arr2):

    # Do quick sanity checks of arguments
    _check_series_args(arr1=arr1, arr2=arr2)

    if isinstance(arr1, pd.Series):
        return pd.Series(_max(arr1.values, arr2.values), index=arr1.index)
    elif isinstance(arr1, np.ndarray):
        return _max(arr1, arr2)


def abs(arr):
    # Do quick sanity checks of arguments
    _check_series_args(arr=arr)

    if isinstance(arr, pd.Series):
        return pd.Series(_abs(arr.values), index=arr.index)
    elif isinstance(arr, np.ndarray):
        return _abs(arr)


def value_when(arr, cond):
    # Do quick sanity checks of arguments
    _check_series_args(arr=arr, cond=cond)

    try:
        if isinstance(cond, pd.Series):
            return pd.Series(_value_when(arr.values, cond.values), index=cond.index)
        elif isinstance(cond, np.ndarray):
            return _value_when(arr, cond)
    except ValueError as exc:
            raise YaUberAlgoInternalError(str(exc))


def nz(arr, fill_by):
    # Do quick sanity checks of arguments
    _check_series_args(arr=arr)

    if not isinstance(fill_by, (float, int, bool, np.float, np.int, np.int32, np.bool)):
        raise YaUberAlgoArgumentError(f"'fill_by' must be float, int, bool, got {type(fill_by)}")

    if not isfinite(fill_by):
        raise YaUberAlgoArgumentError(f"'fill_by' must be finite number, got {fill_by}")

    if isinstance(arr, pd.Series):
        return pd.Series(_nz(arr.values, fill_by), index=arr.index)
    elif isinstance(arr, np.ndarray):
        return _nz(arr, fill_by)


def quantile(arr, period, q):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    _q = _get_series_or_number(q, arr)
    is_not_nan_q = ~np.isnan(_q)

    if not np.all(np.logical_and(_q[is_not_nan_q] >= 0, _q[is_not_nan_q] <= 1.0)):
        raise YaUberAlgoArgumentError("'q' parameter is out of bounds, q >= 0 and q <= 1.0, or NaN for skipped values")

    # Do quick sanity checks of arguments
    _check_series_args(arr=arr, q=_q)

    if arr.dtype == np.bool:
        raise YaUberAlgoDtypeNotSupportedError("Boolean dtype is not supported")

    try:
        if isinstance(arr, pd.Series):
            return pd.Series(_quantile(arr.values, period, _q.values), index=arr.index)
        elif isinstance(arr, np.ndarray):
            return _quantile(arr, period, _q)
    except ValueError as exc:
            raise YaUberAlgoInternalError(str(exc))


def median(arr, period):
    return quantile(arr, period, 0.5)


def roc(arr, period):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _check_series_args(arr=arr)

    if arr.dtype == np.bool:
        raise YaUberAlgoDtypeNotSupportedError("Boolean dtype is not supported")

    try:
        if isinstance(arr, pd.Series):
            return pd.Series(_roc(arr.values, period), index=arr.index)
        elif isinstance(arr, np.ndarray):
            return _roc(arr, period)
    except ValueError as exc:
            raise YaUberAlgoInternalError(str(exc))


def roc_log(arr, period):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _check_series_args(arr=arr)

    if arr.dtype == np.bool:
        raise YaUberAlgoDtypeNotSupportedError("Boolean dtype is not supported")

    try:
        if isinstance(arr, pd.Series):
            return pd.Series(_roc_log(arr.values, period), index=arr.index)
        elif isinstance(arr, np.ndarray):
            return _roc_log(arr, period)
    except ValueError as exc:
            raise YaUberAlgoInternalError(str(exc))


def diff(arr, period):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _check_series_args(arr=arr)

    if arr.dtype == np.bool:
        raise YaUberAlgoDtypeNotSupportedError("Boolean dtype is not supported")

    if isinstance(arr, pd.Series):
        return pd.Series(_diff(arr.values, period), index=arr.index)
    elif isinstance(arr, np.ndarray):
        return _diff(arr, period)


def rsi(arr, period):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _check_series_args(arr=arr)

    if arr.dtype == np.bool:
        raise YaUberAlgoDtypeNotSupportedError("Boolean dtype is not supported")

    if isinstance(arr, pd.Series):
        return pd.Series(_rsi(arr.values, period), index=arr.index)
    elif isinstance(arr, np.ndarray):
        return _rsi(arr, period)


def rangehilo(o, h, l, c, period):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _check_series_args(o=o, h=h, l=l, c=c)

    if o.dtype == np.bool or h.dtype == np.bool or l.dtype == np.bool or c.dtype == np.bool:
        raise YaUberAlgoDtypeNotSupportedError("Boolean dtype is not supported")

    try:
        if isinstance(c, pd.Series):
            return pd.Series(_rangehilo(o.values, h.values, l.values, c.values, period), index=c.index)
        elif isinstance(c, np.ndarray):
            return _rangehilo(o, h, l, c, period)
    except ValueError as exc:
            raise YaUberAlgoInternalError(str(exc))


def rangeclose(h, l, c, period):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _check_series_args(h=h, l=l, c=c)

    if h.dtype == np.bool or l.dtype == np.bool or c.dtype == np.bool:
        raise YaUberAlgoDtypeNotSupportedError("Boolean dtype is not supported")

    try:
        if isinstance(c, pd.Series):
            return pd.Series(_rangeclose(h.values, l.values, c.values, period), index=c.index)
        elif isinstance(c, np.ndarray):
            return _rangeclose(h, l, c, period)
    except ValueError as exc:
        raise YaUberAlgoInternalError(str(exc))


def wma(arr, weight, period):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _check_series_args(arr=arr, weight=weight)
    try:
        if isinstance(arr, pd.Series):
            return pd.Series(_wma(arr.values, weight.values, period), index=arr.index)
        elif isinstance(arr, np.ndarray):
            return _wma(arr, weight, period)
    except ValueError as exc:
        raise YaUberAlgoInternalError(str(exc))


def correlation(x, y, period):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _check_series_args(x=x, y=y)
    try:
        if isinstance(x, pd.Series):
            return pd.Series(_correlation(x.values, y.values, period), index=x.index)
        elif isinstance(x, np.ndarray):
            return _correlation(x, y, period)
    except ValueError as exc:
        raise YaUberAlgoInternalError(str(exc))


def categorize(arr, bins):
    if len(bins) > 2:
        try:
            _bins = np.array(bins, dtype=np.float)
        except ValueError:
            raise YaUberAlgoArgumentError("'bins' must be array of numbers")

        if not np.all(_bins == np.unique(_bins)):
            raise YaUberAlgoArgumentError("'bins' must be unique and sorted in ascending order")
        if not np.all(np.isfinite(_bins)):
            raise YaUberAlgoArgumentError("'bins' include NaN or Inf numbers")
    else:
        raise YaUberAlgoArgumentError("'bins' must include at least 3 elements: [min, bin1..binX, max]")

    # Do quick sanity checks of arguments
    _check_series_args(arr=arr)

    if arr.dtype == np.bool:
        raise YaUberAlgoDtypeNotSupportedError("np.bool dtype is not supported")

    try:
        if isinstance(arr, pd.Series):
            return pd.Series(_categorize(arr.values, _bins), index=arr.index)
        elif isinstance(arr, np.ndarray):
            return _categorize(arr, _bins)
    except ValueError as exc:
        raise YaUberAlgoInternalError(str(exc))


def truerange(h, l, c, period, is_pct=False):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _check_series_args(h=h, l=l, c=c)

    if h.dtype == np.bool or l.dtype == np.bool or c.dtype == np.bool:
        raise YaUberAlgoDtypeNotSupportedError("Boolean dtype is not supported")

    try:
        if isinstance(c, pd.Series):
            return pd.Series(_truerange(h.values, l.values, c.values, period, is_pct), index=c.index)
        elif isinstance(c, np.ndarray):
            return _truerange(h, l, c, period, is_pct)
    except ValueError as exc:
        raise YaUberAlgoInternalError(str(exc))


def updn_ratio(arr, period):
    if period <= 0:
        raise YaUberAlgoArgumentError(f"'period' must be positive number")

    # Do quick sanity checks of arguments
    _check_series_args(arr=arr)

    if arr.dtype == np.bool:
        raise YaUberAlgoDtypeNotSupportedError("Boolean dtype is not supported")

    if isinstance(arr, pd.Series):
        return pd.Series(_updn_ratio(arr.values, period), index=arr.index)
    elif isinstance(arr, np.ndarray):
        return _updn_ratio(arr, period)