import unittest
from yauber_algo.errors import *


class BarsSinceTestCase(unittest.TestCase):
    def test_bars_since(self):
        import yauber_algo.sanitychecks as sc

        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import bars_since

        #
        # Function settings
        #
        algo = 'bars_since'
        func = bars_since

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                array([nan, nan, 0, 1, 0, 0], dtype=np.float),
                func, (
                    array([0, 0, 1, 0, 1, 1], dtype=np.float),  # cond
                ),
                suffix='cond'
            )

            s.check_regular(
                array([nan, nan, 0, 1, nan, 0], dtype=np.float),
                func, (
                    array([0, 0, 1, 0, nan, 1], dtype=np.float),  # cond
                ),
                suffix='cond_nan'
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, nan], dtype=np.float),
                func, (
                    array([0, 0, 0, 0, nan, nan], dtype=np.float),  # cond
                ),
                suffix='cond_nan2'
            )

            s.check_naninf(
                array([nan, nan, nan, nan, nan, nan], dtype=np.float),
                func, (
                    array([0, 0, 0, inf, nan, nan], dtype=np.float),  # cond
                ),
                suffix='codex20180209'
            )


            s.check_regular(
                array([nan, nan, nan, nan, nan, nan], dtype=np.float),
                func, (
                    array([0, inf, 2, 0, nan, nan], dtype=np.float),  # cond
                ),
                suffix='cond_non_binary_raises',
                exception=YaUberAlgoInternalError,
            )

            s.check_series(
                pd.Series(array([nan, nan, 0, 1, 0, 0], dtype=np.float)),
                func, (
                    pd.Series(array([0, 0, 1, 0, 1, 1], dtype=np.float)),  # cond
                ),
                suffix='cond'
            )

            s.check_dtype_float(
                array([nan, nan, 0, 1, 0, 0], dtype=np.float),
                func, (
                    array([0, 0, 1, 0, 1, 1], dtype=np.float),  # cond
                ),
                suffix='cond'
            )

            s.check_dtype_bool(
                array([nan, nan, 0, 1, 0, 0], dtype=np.float),
                func, (
                    array([0, 0, 1, 0, 1, 1], dtype=np.bool),  # cond
                ),
                suffix='cond'
            )

            s.check_dtype_int(
                array([nan, nan, 0, 1, 0, 0], dtype=np.float),
                func, (
                    array([0, 0, 1, 0, 1, 1], dtype=np.int32),  # cond
                ),
                suffix='cond'
            )

            s.check_dtype_object(
                func, (
                    array([0, 0, 1, 0, 1, 1], dtype=np.object),  # cond
                ),
                suffix='cond'
            )

            s.check_futref(5, 1,
                           func, (
                               array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1], dtype=np.float),  # cond
                           ),
                           min_checks_count=3,
                           )

            s.check_window_consistency(5, 1,
                                       func, (
                                           array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1], dtype=np.float),  # cond
                                       ),
                                       min_checks_count=3,
                                       )




