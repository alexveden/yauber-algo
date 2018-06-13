import unittest
from yauber_algo.errors import *


class IIFTestCase(unittest.TestCase):
    def test_iif(self):
        import yauber_algo.sanitychecks as sc

        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import iif

        #
        # Function settings
        #
        algo = 'iif'
        func = iif

        setattr(sys.modules[func.__module__], 'IS_WARN_FUTREF', False)
        setattr(sys.modules[func.__module__], 'IS_RAISE_FUTREF', False)

        cond = array([0, 1, 0, 1, 0, 1], dtype=np.float)
        cond_nan = array([0, 1, 0, 1, nan, 1], dtype=np.float)
        cond_non_bin = array([0, 1, 0, 2, 0, 1], dtype=np.float)
        cond_bool = array([False, True, False, True, False, True], dtype=np.bool)
        cond_object = array([False, True, False, True, False, nan], dtype=np.object)
        cond_nan_inf = array([0, inf, 0, 1, nan, 1], dtype=np.float)

        cond_int32 = array([0, 1, 0, 1, 0, 1], dtype=np.int32)

        arr_true = array([1, 2, 3, 4, 5, 6], dtype=np.float)
        arr_false = array([-1, -2, -3, -4, -5, -6], dtype=np.float)

        arr_true_int32 = array([1, 2, 3, 4, 5, 6], dtype=np.int32)
        arr_false_int32 = array([-1, -2, -3, -4, -5, -6], dtype=np.int32)

        arr_true_bool = array([True, True, True, True, True, True], dtype=np.bool)
        arr_false_bool = array([False, False, False, False, False, False], dtype=np.bool)

        arr_true_nan_inf = array([1, 2, 3, 4, 5, inf], dtype=np.float)
        arr_false_nan_inf = array([inf, -2, -3, -4, -5, -6], dtype=np.float)


        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                array([-1, 2, -3, 4, -5, 6], dtype=np.float),
                func, (cond, arr_true, arr_false),
                suffix='cond'
            )

            s.check_regular(
                array([-1, 2, -3, 4, nan, 6], dtype=np.float),
                func, (cond_nan, arr_true, arr_false),
                suffix='cond_nan'
            )

            s.check_regular(
                array([-1, 2, -3, 4, nan, 6], dtype=np.float),
                func, (cond_non_bin, arr_true, arr_false),
                suffix='cond_non_bin_exception',
                exception=YaUberAlgoInternalError,
            )

            s.check_regular(
                array([-1, 2, -3, 4, -5, 6], dtype=np.float),
                func, (cond_bool, arr_true, arr_false),
                suffix='cond_bool',
            )

            s.check_regular(
                array([-1, 2, -3, 4, -5, 6], dtype=np.float),
                func, (cond_object, arr_true, arr_false),
                suffix='cond_object',
                exception=YaUberAlgoDtypeNotSupportedError,
            )

            s.check_regular(
                array([-1, 2, -3, 4, -5, 6], dtype=np.float),
                func, (cond_bool[:2], arr_true, arr_false),
                suffix='cond_diff_length1',
                exception=YaUberAlgoArgumentError,
            )

            s.check_regular(
                array([-1, 2, -3, 4, -5, 6], dtype=np.float),
                func, (cond_bool, arr_true, arr_false[:2]),
                suffix='cond_diff_length2',
                exception=YaUberAlgoArgumentError,
            )

            s.check_naninf(
                array([nan, nan, -3, 4, nan, nan], dtype=np.float),
                func, (
                    array([0, inf, 0, 1, nan, 1], dtype=np.float),
                    arr_true_nan_inf,
                    arr_false_nan_inf
                ),
            )

            s.check_regular(
                array([-1, 1, -3, 1, -5, 1], dtype=np.float),
                func, (cond, 1, arr_false),
                suffix='cond_true_is_number'
            )

            s.check_regular(
                array([-2, 1, -2, 1, -2, 1], dtype=np.float),
                func, (cond, 1, -2),
                suffix='cond_false_is_number'
            )

            s.check_regular(
                array([-2, 1, -2, 1, -2, 1], dtype=np.float),
                func, (cond, 1, pd.Series(arr_false)),
                suffix='different_types',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([-2, 1, -2, 1, -2, 1], dtype=np.float),
                func, ([1, 2, 3], 1, pd.Series(arr_false)),
                suffix='cond_different_types',
                exception=YaUberAlgoArgumentError
            )

            s.check_series(
                pd.Series(array([-1, 2, -3, 4, -5, 6], dtype=np.float)),
                func, (pd.Series(cond), pd.Series(arr_true), pd.Series(arr_false)),
                suffix=''
            )

            s.check_series(
                pd.Series(array([-1, 1, -3, 1, -5, 1], dtype=np.float)),
                func, (pd.Series(cond), 1, pd.Series(arr_false)),
                suffix='cond_true_is_number'
            )

            s.check_series(
                pd.Series(array([-2, 1, -2, 1, -2, 1], dtype=np.float)),
                func, (pd.Series(cond), 1, -2),
                suffix='cond_false_is_number'
            )

            s.check_dtype_float(
                array([-1, 2, -3, 4, -5, 6], dtype=np.float),
                func, (cond, arr_true, arr_false),
            )

            s.check_dtype_bool(
                array([0, 1, 0, 1, 0, 1], dtype=np.float),
                func, (cond_bool, arr_true_bool, arr_false_bool),
            )

            s.check_dtype_object(
                func, (cond_object, arr_true_bool, arr_false_bool),
            )

            s.check_dtype_int(
                array([-1, 2, -3, 4, -5, 6], dtype=np.float),
                func, (cond_int32, arr_true_int32, arr_false_int32),
            )

            s.check_futref(3, 1,
                func, (cond_int32, arr_true_int32, arr_false_int32),
                min_checks_count=3,
            )

            s.check_window_consistency(3, 1,
                           func, (cond_int32, arr_true_int32, arr_false_int32),
                           min_checks_count=3,
                           )






