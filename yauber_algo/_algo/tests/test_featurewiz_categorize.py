import unittest
from yauber_algo.errors import *


class CategorizeTestCase(unittest.TestCase):
    def test_categorize(self):
        import yauber_algo.sanitychecks as sc
        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import categorize

        #
        # Function settings
        #
        algo = 'categorize'
        func = categorize

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                np.array([0., 0., 0., 0., 1., 1., 1., 2., 2., 2.]),
                func,
                (
                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10]),
                    [0, 3, 6, 10]
                ),
                suffix='reg'
            )

            s.check_regular(
                np.array([0., 0., 0., 0., 1., 1., 1., 2., 2., 2.]),
                func,
                (
                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10]),
                    [0.1, 3, 6, 10]
                ),
                suffix='min_not_in_bins',
                exception=YaUberAlgoInternalError
            )

            s.check_regular(
                np.array([0., 0., 0., 0., 1., 1., 1., 2., 2., 2.]),
                func,
                (
                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10]),
                    [0, 3, 6, 9.999]
                ),
                suffix='max_not_in_bins',
                exception=YaUberAlgoInternalError
            )

            s.check_regular(
                np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                func,
                (
                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10]),
                    [0, 10]
                ),
                suffix='min_max_one_bin',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                func,
                (
                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10]),
                    [0, 10, 10]
                ),
                suffix='bins_non_unique',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                func,
                (
                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10]),
                    [0, 10, 5]
                ),
                suffix='bins_not_sorted',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                func,
                (
                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10]),
                    [0, 5, 'obj']
                ),
                suffix='bins_non_number',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                func,
                (
                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10]),
                    [0, 5, nan]
                ),
                suffix='bins_nan',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                func,
                (
                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10]),
                    [0, 5, inf]
                ),
                suffix='bins_inf',
                exception=YaUberAlgoArgumentError
            )

            s.check_naninf(
                np.array([0., 0., 0., 0., 1., 1., 1., 2., nan, nan]),
                func,
                (
                    np.array([0, 1, 2, 3, 4, 5, 6, 7, inf, nan]),
                    [0, 3, 6, 10]
                ),
                suffix='reg'
            )

            s.check_regular(
                np.array([0., 0., 0., 0., 1., 1., 1., 2., 2., 2.]),
                func,
                (
                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10]),
                    np.array([0, 3, 6, 10])
                ),
                suffix='bins_are_np_array'
            )

            s.check_regular(
                np.array([0., 0., 0., 0., 1., 1., 1., 2., 2., 2.]),
                func,
                (
                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10]),
                    pd.Series([0, 3, 6, 10])
                ),
                suffix='bins_are_series'
            )

            s.check_series(
                pd.Series(np.array([0., 0., 0., 0., 1., 1., 1., 2., 2., 2.])),
                func,
                (
                    pd.Series(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10])),
                    [0, 3, 6, 10]
                ),
                suffix=''
            )

            s.check_dtype_float(
                np.array([0., 0., 0., 0., 1., 1., 1., 2., 2., 2.], dtype=np.float),
                func,
                (
                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10], dtype=np.float),
                    [0, 3, 6, 10]
                ),
                suffix=''
            )

            s.check_dtype_int(
                np.array([0., 0., 0., 0., 1., 1., 1., 2., 2., 2.], dtype=np.float),
                func,
                (
                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10], dtype=np.int32),
                    [0, 3, 6, 10]
                ),
                suffix=''
            )

            s.check_dtype_bool(
                np.array([0., 0., 0., 0., 1., 1., 1., 2., 2., 2.], dtype=np.float),
                func,
                (
                    np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1], dtype=np.bool),
                    [0, 3, 6, 10]
                ),
                suffix='',
                exception=YaUberAlgoDtypeNotSupportedError
            )

            s.check_dtype_object(
                func,
                (
                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10], dtype=np.object),
                    [0, 3, 6, 10]
                ),
                suffix=''
            )

            s.check_futref(5, 1,
                           func,
                           (
                               np.random.random(1000),
                               [0, 0.33, 0.66, 1.0]
                           ),
                           )

            s.check_window_consistency(5, 1,
                                       func,
                                       (
                                           np.random.random(1000),
                                           [0, 0.33, 0.66, 1.0]
                                       ),
                                       )