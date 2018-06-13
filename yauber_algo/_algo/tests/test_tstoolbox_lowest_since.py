import unittest
from yauber_algo.errors import *


class LowestSinceTestCase(unittest.TestCase):
    def test_lowest_since(self):
        import yauber_algo.sanitychecks as sc

        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import lowest_since

        #
        # Function settings
        #
        algo = 'lowest_since'
        func = lowest_since

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                array([nan, 2, 2, 4, 4, 6], dtype=np.float),
                func, (
                    array([1, 2, 3, 4, 5, 6], dtype=np.float),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.float),  # cond

                ),
                suffix='cond_increasing'
            )

            s.check_regular(
                array([nan, 5, 4, 3, 2, 1], dtype=np.float),
                func, (
                    array([6, 5, 4, 3, 2, 1], dtype=np.float),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.float),  # cond

                ),
                suffix='cond_decreasing'
            )

            s.check_regular(
                array([nan, 5, 4, nan, 2, 1], dtype=np.float),
                func, (
                    array([6, 5, 4, 3, 2, 1], dtype=np.float),  # arr
                    array([0, 1, 0, nan, 0, 1], dtype=np.float),  # cond

                ),
                suffix='cond_nan'
            )

            s.check_regular(
                array([nan, 2, 2, nan, 2, 6], dtype=np.float),
                func, (
                    array([1, 2, 3, 4, 5, 6], dtype=np.float),  # arr
                    array([0, 1, 0, nan, 0, 1], dtype=np.float),  # cond

                ),
                suffix='cond_incr'
            )

            s.check_regular(
                array([nan, 2, 2, nan, 2, 6], dtype=np.float),
                func, (
                    array([1, 2, 3, nan, 5, 6], dtype=np.float),  # arr
                    array([0, 1, 0, nan, 0, 1], dtype=np.float),  # cond

                ),
                suffix='cond_arr_nan'
            )

            s.check_regular(
                array([nan, 2, 2, nan, 5, 6], dtype=np.float),
                func, (
                    array([1, 2, 3, nan, 5, 6], dtype=np.float),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.float),  # cond

                ),
                suffix='arr_nan_at_cond'
            )

            s.check_regular(
                array([nan, nan, 3, nan, 5, 6], dtype=np.float),
                func, (
                    array([nan, nan, 3, nan, 5, 6], dtype=np.float),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.float),  # cond

                ),
                suffix='arr_nan_at_cond2'
            )

            s.check_regular(
                array([nan, 2, 2, 1, nan, nan], dtype=np.float),
                func, (
                    array([1, 2, 3, 1, nan, nan], dtype=np.float),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.float),  # cond

                ),
                suffix='arr_nan_skipped'
            )

            s.check_regular(
                array([nan, 2, 2, nan, 2, 6], dtype=np.float),
                func, (
                    array([1, 2, 3, 0, 3, 6], dtype=np.float),  # arr
                    array([0, 1, 0, nan, 0, 1], dtype=np.float),  # cond
                ),
                suffix='cond_nan_skipped'
            )

            s.check_regular(
                array([nan, 2, 3, 1, 1, nan], dtype=np.float),
                func, (
                    array([1, inf, -inf, 1, nan, nan], dtype=np.float),  # arr
                    array([0, 1, inf, 1, 0, 1], dtype=np.float),  # cond

                ),
                suffix='cond_non_bool_inf',
                exception=YaUberAlgoInternalError,
            )
            s.check_regular(
                array([nan, 2, 3, 1, 1, nan], dtype=np.float),
                func, (
                    array([1, inf, -inf, 1, nan, nan], dtype=np.float),  # arr
                    array([0, 1, 0, 1, 2, 1], dtype=np.float),  # cond

                ),
                suffix='cond_non_binary',
                exception=YaUberAlgoInternalError,

            )

            s.check_naninf(
                array([nan, nan, nan, 1, nan, nan], dtype=np.float),
                func, (
                    array([1, inf, -inf, 1, nan, nan], dtype=np.float),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.float),  # cond

                ),
                suffix='inf_replaced'
            )

            s.check_naninf(
                array([nan, nan, 2, nan, nan, nan], dtype=np.float),
                func, (
                    array([1, inf, 2, inf, nan, nan], dtype=np.float),  # arr
                    array([0, 1, 0, 0, 0, 1], dtype=np.float),  # cond

                ),
                suffix='number_is_greater'
            )

            s.check_naninf(
                array([nan, nan, 2, nan, nan, nan], dtype=np.float),
                func, (
                    array([nan, inf, 2, 3, 4, 5], dtype=np.float),  # arr
                    array([0, 1, 0, nan, nan, nan], dtype=np.float),  # cond

                ),
                suffix='cond_nan_replaced'
            )

            s.check_naninf(
                array([nan, nan, 2, nan, nan, nan], dtype=np.float),
                func, (
                    array([1, -inf, 2, -inf, nan, nan], dtype=np.float),  # arr
                    array([0, 1, 0, 0, 0, 1], dtype=np.float),  # cond

                ),
                suffix='number_is_greater_neg'
            )

            s.check_series(
                pd.Series(array([nan, 2, 2, 4, 4, 6], dtype=np.float)),
                func, (
                    pd.Series(array([1, 2, 3, 4, 5, 6], dtype=np.float)),  # arr
                    pd.Series(array([0, 1, 0, 1, 0, 1], dtype=np.float)),  # cond

                ),
                suffix='ser'
            )

            s.check_dtype_float(
                array([nan, 2, 2, 4, 4, 6], dtype=np.float),
                func, (
                    array([1, 2, 3, 4, 5, 6], dtype=np.float),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.float),  # cond

                ),
                suffix=''
            )

            s.check_dtype_int(
                array([nan, 2, 2, 4, 4, 6], dtype=np.float),
                func, (
                    array([1, 2, 3, 4, 5, 6], dtype=np.int32),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.int32),  # cond

                ),
                suffix=''
            )

            s.check_dtype_bool(
                array([nan, 2, 2, 4, 4, 6], dtype=np.float),
                func, (
                    array([1, 2, 3, 4, 5, 6], dtype=np.int32),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.bool),  # cond

                ),
                suffix=''
            )

            s.check_dtype_bool(
                array([nan, 1, 1, 0, 0, 1], dtype=np.float),
                func, (
                    array([1, 1, 1, 0, 1, 1], dtype=np.bool),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.bool),  # cond

                ),
                suffix='both_bool'
            )

            s.check_dtype_object(
                func, (
                    array([1, 1, 1, 0, 1, 1], dtype=np.bool),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.object),  # cond

                ),
            )

            s.check_futref(5, 1,
                           func, (
                               array([1, 2, 3, 4, 5, 6, 0, 0, 0, 1], dtype=np.float),  # arr
                               array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1], dtype=np.float),  # cond

                           ),
                           min_checks_count=3,
                           )

            s.check_window_consistency(5, 1,
                                       func, (
                                           array([1, 2, 3, 4, 5, 6, 0, 0, 0, 1], dtype=np.float),  # arr
                                           array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1], dtype=np.float),  # cond

                                       ),
                                       min_checks_count=3,
                                       )




