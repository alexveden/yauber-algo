import numpy as np
from math import isfinite, isnan, floor, ceil
from numpy import nan
from numba import jit


def cat_sort_unique(cat_arr):
    """
    WARNING: this function must not be called from @jit(nopython=True) code, it will raise the error
    :param cat_arr:
    :return:
    """
    _arr_unique = np.unique(cat_arr)
    return _arr_unique[np.isfinite(_arr_unique)]


@jit(nopython=True)
def cat_get_indexes(sorted_cat_arr, cat_arr):
    len_cat = len(sorted_cat_arr)
    cat_idxs = np.searchsorted(sorted_cat_arr, cat_arr)

    result = np.full(len(cat_idxs), -1.0)
    for i in range(len(cat_idxs)):
        idx = cat_idxs[i]

        if idx < len_cat and sorted_cat_arr[idx] == cat_arr[i]:
            result[i] = float(idx)

    return result


@jit(nopython=True)
def insert_sorted_inplace(arr, new_item, old_item):
    bc = len(arr)
    nan_cnt = 0

    _new_item = nan if not isfinite(new_item) else new_item
    _old_item = nan if not isfinite(old_item) else old_item

    _arr_old_copy = np.copy(arr)

    if isnan(_old_item):
        if isfinite(arr[bc-1]):
            raise ValueError('<old_item> NaN not found at last element')

    if _new_item == _old_item or (isnan(_old_item) and isnan(_new_item)):
        # Elements are equal go fast way
        for i in range(bc):
            if not isfinite(arr[i]):
                nan_cnt += 1

        return nan_cnt

    if _new_item < _old_item or isnan(_old_item):
        offset = 1
        i = 0
        j = 0
    else:
        offset = -1
        i = bc - 1
        j = bc - 1

    new_inserted = False
    old_deleted = False
    _new_item_result_idx = -1
    if isnan(_new_item):
        new_inserted = True
        _new_item_result_idx = bc - 1
        j += offset

    while i < bc and i >= 0:
        if offset > 0:
            if not new_inserted and (_new_item < _arr_old_copy[i] or not isfinite(_arr_old_copy[i])):
                _new_item_result_idx = j
                j += offset
                new_inserted = True

            if not old_deleted and _old_item == _arr_old_copy[i]:
                i += offset
                old_deleted = True
        else:
            if not new_inserted and _new_item > _arr_old_copy[i]:
                _new_item_result_idx = j
                j += offset
                new_inserted = True

            if not old_deleted and _old_item == _arr_old_copy[i]:
                i += offset
                old_deleted = True

        if j >= bc or i >= bc or j < 0 or i < 0:
            break

        arr[j] = _arr_old_copy[i]

        if not isfinite(arr[j]):
            nan_cnt += 1

        j += offset
        i += offset

    arr[_new_item_result_idx] = _new_item

    if isnan(_new_item):
        nan_cnt += 1

    if not old_deleted and isfinite(_old_item):
        raise ValueError('<old_item> not found in the array')

    return nan_cnt

@jit(nopython=True)
def shift_and_fill_inplace(arr, fill_last):
    bc = len(arr)
    non_finite_cnt = 0
    for i in range(1, bc):
        arr[i - 1] = arr[i]
        if not isfinite(arr[i]):
            non_finite_cnt += 1

    arr[-1] = fill_last
    if not isfinite(fill_last):
        non_finite_cnt += 1

    return non_finite_cnt


@jit(nopython=True)
def fill_na_inplace(arr, by_value):
    bc = len(arr)
    for i in range(bc):
        if not isfinite(arr[i]):
            arr[i] = by_value


@jit(nopython=True)
def _percent_rank(arr, period):
    """
    Returns the percent rank of LAST element of a
    :param arr: array-like
    :return: percent rank
    """
    MIN_PERIODS = 5

    if period < MIN_PERIODS:
        raise ValueError('Period must be >= 5')

    bc = len(arr)
    result = np.full(bc, nan)

    for i in range(period, bc):
        k = i - 1
        gtcount = 0
        eqcount = 0
        cnt = 0
        iend = i - period

        if not isfinite(arr[i]):
            continue

        while k >= iend:
            if isfinite(arr[k]):
                if arr[i] > arr[k]:
                    gtcount += 1
                if arr[i] == arr[k]:
                    eqcount += 1
                cnt += 1
            k -= 1

        if cnt >= MIN_PERIODS:
            # Calculate average rank between min / max (this helps to handle special cases with categorical data)
            # Like arr = [5,5,5,5,5,5]:
            #   - if arr[i] > arr[k]: -> result will be 0
            #   - if arr[i] >= arr[k]: -> result will be 1.0
            #   - (2*gtcount+eqcount) / cnt / 0.02 - returns 0.5 (this makes sense)

            # (gtcount / cnt + (gtcount + eqcount) / cnt) / 2.0  => (2 * gt + eq) / cnt / 2.0 (30% faster!!!)
            result[i] = (2*gtcount+eqcount) / cnt / 2.0

    return result


@jit(nopython=True)
def _percent_rank_category(arr, category, sorted_cat_arr, period):
    MIN_PERIODS = 5

    if period < MIN_PERIODS:
        raise ValueError('Period must be >= 5')

    if len(sorted_cat_arr) > 100:
        raise ValueError("Too many categories, no more than 100 allowed")

    bc = len(arr)
    result = np.full(bc, nan)

    cat_indexes = cat_get_indexes(sorted_cat_arr, category)

    # Create matrix of CatIDX x PeriodWindow for category values
    cat_buff = np.full((len(sorted_cat_arr), period+1), nan)

    for i in range(bc):
        cat = cat_indexes[i]

        if not isfinite(cat) or cat < 0:  # Skip NaN categories or that are not found
            continue

        _cat_values = cat_buff[int(cat)]

        # Fill array values in place
        shift_and_fill_inplace(_cat_values, arr[i])

        if not isfinite(arr[i]):
            continue

        k = period - 1
        gtcount = 0
        eqcount = 0
        cnt = 0

        while k >= 0:
            if isfinite(_cat_values[k]):
                if arr[i] > _cat_values[k]:
                    gtcount += 1
                if arr[i] == _cat_values[k]:
                    eqcount += 1
                cnt += 1
            k -= 1

        if cnt >= MIN_PERIODS:
            # Calculate average rank between min / max (this helps to handle special cases with categorical data)
            # Like arr = [5,5,5,5,5,5]:
            #   - if arr[i] > arr[k]: -> result will be 0
            #   - if arr[i] >= arr[k]: -> result will be 1.0
            #   - (2*gtcount+eqcount) / cnt / 0.02 - returns 0.5 (this makes sense)

            # (gtcount / cnt + (gtcount + eqcount) / cnt) / 2.0  => (2 * gt + eq) / cnt / 2.0 (30% faster!!!)
            result[i] = (2 * gtcount + eqcount) / cnt / 2.0

    return result


# @jit is not applicable because of func argument
def _apply_rolling(arr, period, func, exclude_nan=True):
    bc = len(arr)
    result = np.full(bc, nan)

    rolling_window_buff = np.full(period, nan)

    for i in range(bc):
        # Fill array values in place
        nan_cnt = shift_and_fill_inplace(rolling_window_buff, arr[i])

        if not isfinite(arr[i]) or i < period-1:
            continue

        if exclude_nan and nan_cnt > 0:
            result[i] = func(rolling_window_buff[np.isfinite(rolling_window_buff)])
        else:
            result[i] = func(rolling_window_buff)

    # Fill all inf, -inf in place by NaN
    fill_na_inplace(result, nan)

    return result


# @jit is not applicable because of func argument
def _apply_categorical(arr, period, func, category, sorted_cat_arr, return_as_cat=None, exclude_nan=True):
    bc = len(arr)

    result = np.full(bc, nan)

    cat_indexes = cat_get_indexes(sorted_cat_arr, category)

    if return_as_cat is not None:
        ret_cat_indexes = cat_get_indexes(sorted_cat_arr, return_as_cat)

    # Create matrix of CatIDX x PeriodWindow for category values
    cat_buff = np.full((len(sorted_cat_arr), period), nan)
    cat_cnts = np.full(len(sorted_cat_arr), 0)

    for i in range(bc):
        cat = cat_indexes[i]
        category_valid = False
        _icat = -1

        if cat >= 0:  # Skip NaN categories or that are not found
            _icat = int(cat)
            _cat_values = cat_buff[_icat]

            # Fill array values in place
            nan_cnt = shift_and_fill_inplace(_cat_values, arr[i])

            cat_cnts[_icat] += 1
            category_valid = isfinite(arr[i])

        if return_as_cat is not None:
            _icat_ret = int(ret_cat_indexes[i])

            if _icat_ret == -1:
                category_valid = False
            else:
                if _icat != _icat_ret:
                    # Ignore ALGOCODEX 2018-02-09 rule: when referencing other periods / categories
                    # But if _icat == _icat_ret - apply this CODEX rule!
                    category_valid = True
                    _icat = _icat_ret
                    _cat_values = cat_buff[_icat]
                    nan_cnt = 1  # Force func(_cat_values[np.isfinite(_cat_values)])

        if not category_valid:
            continue

        if cat_cnts[_icat] < period:
            continue

        if exclude_nan and nan_cnt > 0:
            result[i] = func(_cat_values[np.isfinite(_cat_values)])
        else:
            result[i] = func(_cat_values)


    return result


@jit(nopython=True)
def _quantile(arr, period, q):
    MIN_COUNT = 5

    if period < MIN_COUNT:
        raise ValueError("period < 5")

    bc = len(arr)
    result = np.full(bc, nan)

    rolling_window_buff = np.full(period, nan)

    for i in range(bc):
        # Fill array values in place
        if i < period:
            nan_cnt = insert_sorted_inplace(rolling_window_buff, arr[i], nan)
        else:
            nan_cnt = insert_sorted_inplace(rolling_window_buff, arr[i], arr[i - period])

        non_nan_cnt = period - nan_cnt
        if isfinite(arr[i]) and isfinite(q[i]) and i >= period-1 and non_nan_cnt >= MIN_COUNT:
            # Check if we have enough data for quantile calculation, and use 'midpoint' method of quantile calculation
            lower_i = floor(q[i] * (non_nan_cnt - 1))
            upper_i = ceil(q[i] * (non_nan_cnt - 1))
            result[i] = (rolling_window_buff[lower_i] + rolling_window_buff[upper_i]) / 2.0

    return result


@jit(nopython=True)
def _categorize(arr, bins):
    bc = len(arr)
    bincnt = len(bins)
    result = np.full(bc, nan)

    for i in range(bc):
        if isfinite(arr[i]):
            ibin = -1
            for j in range(bincnt):
                if arr[i] <= bins[j]:
                    if j == 0 and arr[i] == bins[j]:
                        ibin = j
                    else:
                        ibin = j - 1
                    break
            if ibin < 0:
                raise ValueError("Array value is out of bins min/max range. Bins must include min/max range of array: "
                                 "example of bins for [0;1.0] series [0, 0.33, 0.66, 1.0]")

            result[i] = ibin

    return result