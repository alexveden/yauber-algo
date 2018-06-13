import numpy as np
from math import isfinite, isnan, sqrt, log
from numpy import nan
from numba import jit


@jit(nopython=True)
def _ref(arr, period):
    bc = len(arr)
    result = np.full(bc, np.nan)

    for i in range(max(0, -period), min(bc, bc - period)):
        if not isfinite(arr[i + period]):
            # Replace all NaN, Inf, -Inf by NaN!
            result[i] = nan
        else:
            result[i] = arr[i + period]

    return result


@jit(nopython=True)
def _iif(cond, if_true_arr, if_false_arr):
    bc = len(cond)
    result = np.full(bc, np.nan)

    for i in range(bc):
        if not isfinite(cond[i]):
            # Replace all NaN, Inf, -Inf by NaN!
            result[i] = nan
        else:
            if cond[i] == 1.0 or cond[i] is True:
                if isfinite(if_true_arr[i]):
                    result[i] = if_true_arr[i]
            elif cond[i] == 0.0 or cond[i] is False:
                if isfinite(if_false_arr[i]):
                    result[i] = if_false_arr[i]
            else:
                raise ValueError("Only True / False / 1.0 / 0.0 and NaN values are allowed for 'cond' array")

    return result


@jit(nopython=True)
def _hhv(arr, period):
    if period <= 0:
        raise ValueError()

    bc = len(arr)
    result = np.full(bc, nan)

    cur = nan
    icur = -1

    for i in range(bc):
        if not isfinite(arr[i]):
            continue

        if arr[i] > cur or not isfinite(cur):
            cur = arr[i]
            icur = i
        else:
            if i - icur >= period:
                cur = nan
                icur = i - period + 1

                for k in range(i - period + 1, i + 1):
                    if not isfinite(arr[k]):
                        continue

                    if k == i - period + 1 or arr[k] > cur or not isfinite(cur):
                        cur = arr[k]
                        icur = k

        if i >= period - 1:
            result[i] = cur

    return result


@jit(nopython=True)
def _llv(arr, period):
    if period <= 0:
        raise ValueError()

    bc = len(arr)
    result = np.full(bc, nan)

    cur = nan
    icur = -1

    for i in range(bc):
        if not isfinite(arr[i]):
            continue

        if arr[i] < cur or not isfinite(cur):
            cur = arr[i]
            icur = i
        else:
            if i - icur >= period:
                cur = nan
                icur = i - period + 1

                for k in range(i - period + 1, i + 1):
                    if not isfinite(arr[k]):
                        continue

                    if k == i - period + 1 or arr[k] < cur or not isfinite(cur):
                        cur = arr[k]
                        icur = k

        if i >= period - 1:
            result[i] = cur

    return result


@jit(nopython=True)
def _highest_since(arr, cond):
    bc = len(arr)
    result = np.full(bc, nan)
    cur = nan
    has_cond = False

    for i in range(bc):
        if cond[i] == 1.0 or cond[i] is True:
            cur = nan
            has_cond = True
            if isfinite(arr[i]):
                cur = arr[i]

        elif cond[i] == 0.0 or cond[i] is False or isnan(cond[i]):
            if not isnan(cond[i]):
                if isfinite(arr[i]) and (arr[i] > cur or not isfinite(cur)) and has_cond:
                    cur = arr[i]
        else:
            raise ValueError("Only True / False / 1.0 / 0.0 and NaN values are allowed for 'cond' array")

        if isfinite(arr[i]) and isfinite(cond[i]):
            result[i] = cur
        else:
            result[i] = nan

    return result


@jit(nopython=True)
def _lowest_since(arr, cond):
    bc = len(arr)
    result = np.full(bc, nan)
    cur = nan
    has_cond = False

    for i in range(bc):
        if cond[i] == 1.0 or cond[i] is True:
            cur = nan
            has_cond = True
            if isfinite(arr[i]):
                cur = arr[i]

        elif cond[i] == 0.0 or cond[i] is False or isnan(cond[i]):
            if not isnan(cond[i]):
                if isfinite(arr[i]) and (arr[i] < cur or not isfinite(cur)) and has_cond:
                    cur = arr[i]
        else:
            raise ValueError("Only True / False / 1.0 / 0.0 and NaN values are allowed for 'cond' array")

        if isfinite(arr[i]) and isfinite(cond[i]):
            result[i] = cur
        else:
            result[i] = nan

    return result


@jit(nopython=True)
def _bars_since(cond):
    bc = len(cond)
    result = np.full(bc, nan)

    icur = -1

    for i in range(bc):
        if not isfinite(cond[i]):
            continue

        if cond[i] == 1.0 or cond[i] is True:
            icur = i
            result[i] = 0
        elif cond[i] == 0.0 or cond[i] is False:
            if icur >= 0:
                result[i] = i - icur
        else:
            raise ValueError("Only True / False / 1.0 / 0.0 and NaN values are allowed for 'cond' array")

    return result


@jit(nopython=True)
def _cross_up(arr, arr_threshold):
    bc = len(arr_threshold)
    result = np.full(bc, nan)

    for i in range(1, bc):
        if isfinite(arr_threshold[i]) and isfinite(arr[i]) and isfinite(arr_threshold[i - 1]) and isfinite(arr[i - 1]):
            if arr[i - 1] < arr_threshold[i - 1] and arr[i] > arr_threshold[i]:
                result[i] = 1.0
            else:
                result[i] = 0.0

    return result


@jit(nopython=True)
def _cross_dn(arr, arr_threshold):
    bc = len(arr_threshold)
    result = np.full(bc, nan)

    for i in range(1, bc):
        if isfinite(arr_threshold[i]) and isfinite(arr[i]) and isfinite(arr_threshold[i - 1]) and isfinite(arr[i - 1]):
            if arr[i - 1] > arr_threshold[i - 1] and arr[i] < arr_threshold[i]:
                result[i] = 1.0
            else:
                result[i] = 0.0

    return result


@jit(nopython=True)
def _sum(arr, period):
    if period <= 0:
        raise ValueError('Period must be positive')

    bc = len(arr)
    result = np.full(bc, nan)

    _sum_total = 0.0
    _valid_cnt = 0

    for i in range(bc):
        if i < period:
            if isfinite(arr[i]):
                _sum_total += arr[i]
                _valid_cnt += 1

                if i == period - 1 and _valid_cnt > 0:
                    result[i] = _sum_total
        else:
            if isfinite(arr[i - period]):
                _sum_total -= arr[i - period]
                _valid_cnt -= 1

            if isfinite(arr[i]):
                _sum_total += arr[i]
                _valid_cnt += 1

                if _valid_cnt > 0:
                    result[i] = _sum_total

    return result


@jit(nopython=True)
def _ma(arr, period):
    if period <= 0:
        raise ValueError('Period must be positive')

    bc = len(arr)
    result = np.full(bc, nan)

    _sum_total = 0.0
    _valid_cnt = 0

    for i in range(bc):
        if i < period:
            if isfinite(arr[i]):
                _sum_total += arr[i]
                _valid_cnt += 1

                if i == period - 1 and _valid_cnt > 0:
                    result[i] = _sum_total / _valid_cnt
        else:
            if isfinite(arr[i - period]):
                _sum_total -= arr[i - period]
                _valid_cnt -= 1

            if isfinite(arr[i]):
                _sum_total += arr[i]
                _valid_cnt += 1

                if _valid_cnt > 0:
                    result[i] = _sum_total / _valid_cnt

    return result


@jit(nopython=True)
def _stdev(arr, period):
    MIN_STDEV_PERIODS = 5

    if period < MIN_STDEV_PERIODS:
        raise ValueError('Period must be >= 5')

    bc = len(arr)
    result = np.full(bc, nan)

    _sum_total = 0.0
    _valid_cnt = 0
    _disp_sum = 0.0

    for i in range(bc):
        if i < period:
            if isfinite(arr[i]):
                _sum_total += arr[i]
                _valid_cnt += 1
                _disp_sum += arr[i] ** 2

                if i == period - 1 and _valid_cnt >= MIN_STDEV_PERIODS:
                    _avg = (_sum_total / _valid_cnt)
                    result[i] = sqrt((_disp_sum / _valid_cnt) - _avg ** 2)
        else:
            if isfinite(arr[i - period]):
                _sum_total -= arr[i - period]
                _disp_sum -= arr[i - period] ** 2
                _valid_cnt -= 1

            if isfinite(arr[i]):
                _sum_total += arr[i]
                _disp_sum += arr[i] ** 2
                _valid_cnt += 1

                if _valid_cnt >= MIN_STDEV_PERIODS:
                    _avg = (_sum_total / _valid_cnt)
                    result[i] = sqrt((_disp_sum / _valid_cnt) - _avg ** 2)

    return result


@jit(nopython=True)
def _sum_since(arr, cond, first_is_zero=False):
    bc = len(cond)
    result = np.full(bc, nan)

    icur = -1
    cur_sum = nan

    for i in range(bc):
        if not isfinite(cond[i]):
            continue

        if cond[i] == 1.0 or cond[i] is True:
            icur = i

            if isfinite(arr[i]):
                if first_is_zero:
                    cur_sum = 0.0
                else:
                    cur_sum = arr[i]
            else:
                cur_sum = nan
        elif cond[i] == 0.0 or cond[i] is False or isnan(cond[i]):
            if not isnan(cond[i]) and isfinite(arr[i]) and icur >= 0:
                if isfinite(cur_sum):
                    cur_sum += arr[i]
                else:
                    cur_sum = arr[i]
        else:
            raise ValueError("Only True / False / 1.0 / 0.0 and NaN values are allowed for 'cond' array")

        if isfinite(arr[i]):
            result[i] = cur_sum

    return result

@jit(nopython=True)
def _zscore(arr, period):
    MIN_STDEV_PERIODS = 5

    if period < MIN_STDEV_PERIODS:
        raise ValueError('Period must be >= 5')

    bc = len(arr)
    result = np.full(bc, nan)

    _sum_total = 0.0
    _valid_cnt = 0
    _disp_sum = 0.0
    _sd = 0.0

    for i in range(bc):
        if i < period:
            if isfinite(arr[i]):
                _sum_total += arr[i]
                _valid_cnt += 1
                _disp_sum += arr[i] ** 2

                if i == period - 1 and _valid_cnt >= MIN_STDEV_PERIODS:
                    _avg = (_sum_total / _valid_cnt)
                    _sd = sqrt((_disp_sum / _valid_cnt) - _avg ** 2)

                    if _sd == 0.0:
                        # in case of no dispersion like for arr [1, 1, 1, 1, 1] the zScore must be 0.0
                        result[i] = 0.0
                    else:
                        result[i] = (arr[i] - _avg) / _sd
        else:
            if isfinite(arr[i - period]):
                _sum_total -= arr[i - period]
                _disp_sum -= arr[i - period] ** 2
                _valid_cnt -= 1

            if isfinite(arr[i]):
                _sum_total += arr[i]
                _disp_sum += arr[i] ** 2
                _valid_cnt += 1

                if _valid_cnt >= MIN_STDEV_PERIODS:
                    _avg = (_sum_total / _valid_cnt)
                    _sd = sqrt((_disp_sum / _valid_cnt) - _avg ** 2)

                    if _sd == 0.0:
                        # in case of no dispersion like for arr [1, 1, 1, 1, 1] the zScore must be 0.0
                        result[i] = 0.0
                    else:
                        result[i] = (arr[i] - _avg) / _sd

    return result


@jit(nopython=True)
def _min(arr1, arr2):

    bc = len(arr1)
    result = np.full(bc, nan)

    for i in range(bc):
        if isfinite(arr1[i]) and isfinite(arr2[i]):
            if arr1[i] < arr2[i]:
                result[i] = arr1[i]
            else:
                result[i] = arr2[i]

    return result


@jit(nopython=True)
def _max(arr1, arr2):

    bc = len(arr1)
    result = np.full(bc, nan)

    for i in range(bc):
        if isfinite(arr1[i]) and isfinite(arr2[i]):
            if arr1[i] > arr2[i]:
                result[i] = arr1[i]
            else:
                result[i] = arr2[i]

    return result


@jit(nopython=True)
def _abs(arr):

    bc = len(arr)
    result = np.full(bc, nan)

    for i in range(bc):
        if isfinite(arr[i]):
            result[i] = abs(arr[i])

    return result

@jit(nopython=True)
def _value_when(arr, cond):
    bc = len(cond)
    result = np.full(bc, nan)
    cond_value = nan

    for i in range(bc):
        if not isfinite(cond[i]):
            continue

        if cond[i] == 1.0 or cond[i] is True:
            if isfinite(arr[i]):
                cond_value = arr[i]
            else:
                cond_value = nan
        elif not(cond[i] == 0.0 or cond[i] is False or isnan(cond[i])):
            raise ValueError("Only True / False / 1.0 / 0.0 and NaN values are allowed for 'cond' array")

        if isfinite(arr[i]):
            result[i] = cond_value

    return result


@jit(nopython=True)
def _nz(arr, fill_by):

    bc = len(arr)
    result = np.full(bc, fill_by)

    for i in range(bc):
        if isfinite(arr[i]):
            result[i] = arr[i]

    return result

@jit(nopython=True)
def _roc(arr, period):
    if period <= 0:
        raise ValueError(f"'period' must be positive number")

    bc = len(arr)
    result = np.full(bc, nan)

    for i in range(period, bc):
        if isfinite(arr[i]) and isfinite(arr[i-period]):
            if arr[i] <= 0.0 or arr[i - period] <= 0.0:
                raise ValueError("% rate-of-change is only applicable to positive time series, got values less or equal to zero!")

            result[i] = (arr[i]/arr[i-period] - 1.0)

    return result

@jit(nopython=True)
def _roc_log(arr, period):
    if period <= 0:
        raise ValueError(f"'period' must be positive number")

    bc = len(arr)
    result = np.full(bc, nan)

    for i in range(period, bc):
        if isfinite(arr[i]) and isfinite(arr[i-period]):
            if arr[i] <= 0.0 or arr[i - period] <= 0.0:
                raise ValueError("% rate-of-change is only applicable to positive time series, got values less or equal to zero!")

            result[i] = log(arr[i]/arr[i-period])

    return result

@jit(nopython=True)
def _diff(arr, period):
    if period <= 0:
        raise ValueError(f"'period' must be positive number")

    bc = len(arr)
    result = np.full(bc, nan)

    for i in range(period, bc):
        if isfinite(arr[i]) and isfinite(arr[i-period]):
            result[i] = (arr[i] - arr[i-period])

    return result


@jit(nopython=True)
def _rsi(arr, period):
    if period <= 0:
        raise ValueError('Period must be positive')

    bc = len(arr)
    result = np.full(bc, nan)

    sumup = 0
    sumdn = 0

    upcnt = 0
    dncnt = 0

    for i in range(1, bc):

        diff = arr[i] - arr[i - 1]

        if i > period:
            # Remove old_diff from the window first (in case if 'diff' is NaN to avoid skipping)
            old_diff = arr[i - period] - arr[i - period - 1]

            if isfinite(old_diff):
                if old_diff > 0:
                    sumup -= old_diff
                    upcnt -= 1
                elif old_diff < 0:
                    sumdn += old_diff
                    dncnt -= 1

        if not isfinite(diff):
            continue

        if diff > 0:
            sumup += diff
            upcnt += 1
        elif diff < 0:
            sumdn -= diff
            dncnt += 1

        if i >= period:
            if upcnt + dncnt > 0:
                avgup = 0.0 if upcnt == 0 else sumup / upcnt
                avgdn = 0.0 if dncnt == 0 else sumdn / dncnt

                rsi = 100.0 * avgup / (avgup + avgdn)

                assert rsi < 101
                assert rsi >= 0

                result[i] = rsi
            else:
                result[i] = 50.0

    return result

@jit(nopython=True)
def _rangehilo(o, h, l, c, period):
    if period <= 0:
        raise ValueError()

    bc = len(c)
    result = np.full(bc, nan)

    cur_hhv = nan
    icur_hhv = -1

    cur_llv = nan
    icur_llv = -1

    for i in range(bc):
        if isfinite(h[i]):
            if h[i] > cur_hhv or not isfinite(cur_hhv):
                cur_hhv = h[i]
                icur_hhv = i
            else:
                if i - icur_hhv >= period:
                    cur_hhv = nan
                    icur_hhv = i - period + 1

                    for k in range(i - period + 1, i + 1):
                        if not isfinite(h[k]):
                            continue

                        if k == i - period + 1 or h[k] > cur_hhv or not isfinite(cur_hhv):
                            cur_hhv = h[k]
                            icur_hhv = k

        if isfinite(l[i]):
            if l[i] < cur_llv or not isfinite(cur_llv):
                cur_llv = l[i]
                icur_llv = i
            else:
                if i - icur_llv >= period:
                    cur_llv = nan
                    icur_llv = i - period + 1

                    for k in range(i - period + 1, i + 1):
                        if not isfinite(l[k]):
                            continue

                        if k == i - period + 1 or l[k] < cur_llv or not isfinite(cur_llv):
                            cur_llv = l[k]
                            icur_llv = k

        if i >= period - 1:
            _o = o[i-period+1]
            _h = cur_hhv
            _l = cur_llv
            _c = c[i]

            if not (isfinite(_o) and isfinite(_c) and isfinite(_h) and isfinite(_l) and isfinite(h[i]) and isfinite(l[i])):
                continue

            # Do sanity checks
            if h[i] < l[i]:
                raise ValueError("Input data error: H < L")

            if c[i] > h[i] or c[i] < l[i]:
                raise ValueError("Input data error: C < L or C > H")

            if o[i] > h[i] or o[i] < l[i]:
                raise ValueError("Input data error: O < L or O > H")

            # Calculate RangeHiLo
            # RangeHilo - is measure of Doji'ness of the candle(period=1) or range
            # 1.0 - means that a candle is exact Doji
            # 0.0 - means that a candle is trending from Open price to Close (where open is min/max and close is min/max price of candle)
            if _h - _l == 0.0:
                # The range candle is like '-' -> it's closer to Doji rather than to trend candle
                # 2018-02-17 But 1.0 is an another extreme case, because of this the algorithm result must be ambigous i.e. = 0.5
                result[i] = 0.5
            else:
                result[i] = ((_h - max(_o, _c)) + (min(_o, _c) - _l)) / (_h - _l)

    return result


@jit(nopython=True)
def _rangeclose(h, l, c, period):
    if period <= 0:
        raise ValueError()

    bc = len(c)
    result = np.full(bc, nan)

    cur_hhv = nan
    icur_hhv = -1

    cur_llv = nan
    icur_llv = -1

    for i in range(bc):
        if isfinite(h[i]):
            if h[i] > cur_hhv or not isfinite(cur_hhv):
                cur_hhv = h[i]
                icur_hhv = i
            else:
                if i - icur_hhv >= period:
                    cur_hhv = nan
                    icur_hhv = i - period + 1

                    for k in range(i - period + 1, i + 1):
                        if not isfinite(h[k]):
                            continue

                        if k == i - period + 1 or h[k] > cur_hhv or not isfinite(cur_hhv):
                            cur_hhv = h[k]
                            icur_hhv = k

        if isfinite(l[i]):
            if l[i] < cur_llv or not isfinite(cur_llv):
                cur_llv = l[i]
                icur_llv = i
            else:
                if i - icur_llv >= period:
                    cur_llv = nan
                    icur_llv = i - period + 1

                    for k in range(i - period + 1, i + 1):
                        if not isfinite(l[k]):
                            continue

                        if k == i - period + 1 or l[k] < cur_llv or not isfinite(cur_llv):
                            cur_llv = l[k]
                            icur_llv = k

        if i >= period - 1:
            _h = cur_hhv
            _l = cur_llv
            _c = c[i]

            if not (isfinite(_c) and isfinite(_h) and isfinite(_l) and isfinite(h[i]) and isfinite(l[i])):
                continue

            # Do sanity checks
            if h[i] < l[i]:
                raise ValueError("Input data error: H < L")

            if c[i] > h[i] or c[i] < l[i]:
                raise ValueError("Input data error: C < L or C > H")

            # Calculate RangeClose
            # val[i] = (pC[i] - cur_llv) / (cur_hhv - cur_llv);
            if _h - _l == 0.0:
                # The range candle is like '-' -> keep neutral indicator value
                result[i] = 0.5
            else:
                result[i] = (_c - _l) / (_h - _l)

    return result


@jit(nopython=True)
def _wma(arr, weight, period):
    if period <= 0:
        raise ValueError('Period must be positive')

    bc = len(arr)
    result = np.full(bc, nan)

    _sum_total = 0.0
    _valid_w = 0.0

    for i in range(bc):
        a = arr[i]
        w = weight[i]

        if i < period:
            if isfinite(a) and isfinite(w):
                if w < 0:
                    raise ValueError('Negative weight is not allowed')

                _sum_total += a * w
                _valid_w += w

                if i == period - 1 and _valid_w > 0:
                    result[i] = _sum_total / _valid_w
        else:
            a_prev = arr[i - period]
            w_prev = weight[i - period]

            if isfinite(a_prev) and isfinite(w_prev):
                _sum_total -= a_prev * w_prev
                _valid_w -= w_prev

            if isfinite(a) and isfinite(w):
                if w < 0:
                    raise ValueError('Negative weight is not allowed')

                _sum_total += a * w
                _valid_w += w

                if _valid_w > 0:
                    result[i] = _sum_total / _valid_w

    return result


@jit(nopython=True)
def _correlation(x, y, period):
    MIN_STDEV_PERIODS = 5

    if period < MIN_STDEV_PERIODS:
        raise ValueError('Period must be >= 5')

    bc = len(x)
    result = np.full(bc, nan)

    avgx = 0.0
    avgy = 0.0
    avgxy = 0.0
    avgx2 = 0.0
    avgy2 = 0.0

    for i in range(bc):
        xv = x[i]
        yv = y[i]

        if i + 1 < period:
            if isfinite(xv) and isfinite(yv):
                avgx += xv
                avgy += yv
                avgxy += xv * yv
                avgx2 += xv ** 2
                avgy2 += yv ** 2

        else:

            if i + 1 > period:
                xpv = x[i - period]
                ypv = y[i - period]
                if isfinite(xpv) and isfinite(ypv):
                    avgx -= xpv
                    avgy -= ypv
                    avgxy -= xpv * ypv
                    avgx2 -= xpv ** 2
                    avgy2 -= ypv ** 2

            if isfinite(xv) and isfinite(yv):
                avgx += xv
                avgy += yv
                avgxy += xv * yv
                avgx2 += xv ** 2
                avgy2 += yv ** 2

                nom = avgxy / period - (avgx / period) * (avgy / period)
                denom = sqrt(avgx2 / period - (avgx / period) ** 2) * sqrt(avgy2 / period - (avgy / period) ** 2)

                result[i] = nom / denom

    return result


@jit(nopython=True)
def _truerange(h, l, c, period, is_pct):
    if period <= 0:
        raise ValueError()

    bc = len(c)
    result = np.full(bc, nan)

    cur_hhv = nan
    icur_hhv = -1

    cur_llv = nan
    icur_llv = -1

    for i in range(bc):
        if isfinite(h[i]):
            if h[i] > cur_hhv or not isfinite(cur_hhv):
                cur_hhv = h[i]
                icur_hhv = i
            else:
                if i - icur_hhv >= period:
                    cur_hhv = nan
                    icur_hhv = i - period + 1

                    for k in range(i - period + 1, i + 1):
                        if not isfinite(h[k]):
                            continue

                        if k == i - period + 1 or h[k] > cur_hhv or not isfinite(cur_hhv):
                            cur_hhv = h[k]
                            icur_hhv = k

        if isfinite(l[i]):
            if l[i] < cur_llv or not isfinite(cur_llv):
                cur_llv = l[i]
                icur_llv = i
            else:
                if i - icur_llv >= period:
                    cur_llv = nan
                    icur_llv = i - period + 1

                    for k in range(i - period + 1, i + 1):
                        if not isfinite(l[k]):
                            continue

                        if k == i - period + 1 or l[k] < cur_llv or not isfinite(cur_llv):
                            cur_llv = l[k]
                            icur_llv = k

        if i >= period:
            _h = cur_hhv
            _l = cur_llv
            _prev_c = c[i-period]
            _c = c[i]

            if not (isfinite(_c) and isfinite(_h) and isfinite(_l) and isfinite(h[i]) and isfinite(l[i]) and isfinite(_prev_c)):
                continue

            # Do sanity checks
            if h[i] < l[i]:
                raise ValueError("Input data error: H < L")

            if c[i] > h[i] or c[i] < l[i]:
                raise ValueError("Input data error: C < L or C > H")

            # Calculate TrueRange
            tr = max(abs(_c - _prev_c), _h - _l)

            if tr == 0.0:
                # If TrueRange is zero always return NaN
                # because this is ambiguous value, and volatility must be always > 0.0
                result[i] = nan
            else:
                if is_pct:
                    if _prev_c <= 0.0:
                        raise ValueError("logarithmic truerange is only applicable to positive time series, got values less or equal to zero!")

                    result[i] = (tr / _prev_c)
                else:
                    result[i] = tr

    return result


@jit(nopython=True)
def _updn_ratio(arr, period):
    if period <= 0:
        raise ValueError('Period must be positive')

    bc = len(arr)
    result = np.full(bc, nan)

    upcnt = 0
    dncnt = 0
    medcnt = 0

    for i in range(1, bc):

        diff = arr[i] - arr[i - 1]

        if i > period:
            # Remove old_diff from the window first (in case if 'diff' is NaN to avoid skipping)
            old_diff = arr[i - period] - arr[i - period - 1]

            if isfinite(old_diff):
                if old_diff > 0:
                    upcnt -= 1
                elif old_diff < 0:
                    dncnt -= 1
                else:
                    medcnt -= 1

        if not isfinite(diff):
            continue

        if diff > 0:
            upcnt += 1
        elif diff < 0:
            dncnt += 1
        else:
            medcnt += 1

        if i >= period:
            if upcnt + dncnt + medcnt > 0:
                result[i] = ((upcnt + dncnt*-1) / (dncnt + upcnt + medcnt) + 1) / 2.0

    return result