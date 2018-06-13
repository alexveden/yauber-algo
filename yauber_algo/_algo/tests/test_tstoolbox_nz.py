import unittest
from yauber_algo.errors import *


class AbsTestCase(unittest.TestCase):
    def test_abs(self):
        import yauber_algo.sanitychecks as sc
        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import nz

        #
        # Function settings
        #
        algo = 'nz'
        func = nz

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                array([3, 2, 2, 4, 3, 2, 1], dtype=np.float),
                func,
                (
                    array([3, 2, 2, 4, 3, 2, 1], dtype=np.float),
                    0.0,
                ),
                suffix='regular'
            )

            s.check_regular(
                array([3, 2, 0.0, 4, 3, 2, 0.0], dtype=np.float),
                func,
                (
                    array([3, 2, nan, 4, 3, 2, inf], dtype=np.float),
                    0.0,
                ),
                suffix='nan_inf'
            )

            s.check_regular(
                array([3, 2, 0.0, 4, 3, 2, 0.0], dtype=np.float),
                func,
                (
                    array([3, 2, nan, 4, 3, 2, inf], dtype=np.float),
                    inf,
                ),
                suffix='fill_by_inf',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([3, 2, 0.0, 4, 3, 2, 0.0], dtype=np.float),
                func,
                (
                    array([3, 2, nan, 4, 3, 2, inf], dtype=np.float),
                    inf,
                ),
                suffix='fill_by_nan',
                exception=YaUberAlgoArgumentError
            )

            s.check_naninf(
                array([3, 2, 0.0, 4, 3, 2, 0.0], dtype=np.float),
                func,
                (
                    array([3, 2, nan, 4, 3, 2, inf], dtype=np.float),
                    0.0,
                ),
                suffix='nan_inf',
                ignore_nan_argument_position_check=True
            )

            s.check_regular(
                array([3, 2, 1.0, 4, 3, 2, 1.0], dtype=np.float),
                func,
                (
                    array([3, 2, nan, 4, 3, 2, inf], dtype=np.float),
                    True,
                ),
                suffix='nan_bool'
            )

            s.check_regular(
                array([3, 2, 1.0, 4, 3, 2, 1.0], dtype=np.float),
                func,
                (
                    array([3, 2, nan, 4, 3, 2, inf], dtype=np.float),
                    1,
                ),
                suffix='nan_int'
            )

            s.check_regular(
                array([3, 2, 1.0, 4, 3, 2, 1.0], dtype=np.float),
                func,
                (
                    array([3, 2, nan, 4, 3, 2, inf], dtype=np.float),
                    'ok',
                ),
                suffix='nan_object',
                exception=YaUberAlgoArgumentError
            )

            s.check_series(
                pd.Series(array([3, 2, 0.0, 4, 3, 2, 0.0], dtype=np.float)),
                func,
                (
                    pd.Series(array([3, 2, nan, 4, 3, 2, inf], dtype=np.float)),
                    0.0,
                ),
                suffix='regular'
            )

            s.check_dtype_float(
                array([3, 2, 2, 4, 3, 2, 0.0], dtype=np.float),
                func,
                (
                    array([3, 2, 2, 4, 3, 2, nan], dtype=np.float),
                    0.0,
                ),
                suffix='regular'
            )

            s.check_dtype_bool(
                array([0, 1, 0, 1, 0, 1, 0.0], dtype=np.float),
                func,
                (
                    array([0, 1, 0, 1, 0, 1, 0], dtype=np.bool),
                    0.0,
                ),
                suffix='regular'
            )

            s.check_dtype_int(
                array([0, 1, 0, 1, 0, 1, 0.0], dtype=np.float),
                func,
                (
                    array([0, 1, 0, 1, 0, 1, 0], dtype=np.int32),
                    0.0,
                ),
                suffix='regular'
            )

            s.check_dtype_object(
                func,
                (
                    array([0, 1, 0, 1, 0, 1, 0], dtype=np.object),
                    0.0,
                ),
                suffix='regular'
            )

            s.check_futref(5, 1,
                           func,
                           (
                               np.random.random(100),
                               0.0
                           ),
                           )

            s.check_window_consistency(5, 1,
                                       func,
                                       (
                                           np.random.random(100),
                                           0.0
                                       ),
                                       )

