import unittest
from yauber_algo.errors import *


class PercentRankCategoryTestCase(unittest.TestCase):
    def test_percent_rank_category(self):
        import yauber_algo.sanitychecks as sc
        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import percent_rank_category, percent_rank

        #
        # Function settings
        #
        algo = 'percent_rank_category'
        func = percent_rank_category

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                array([nan, nan, nan, nan, nan, .30, .10]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    array([1, 1, 1, 1, 1, 1, 1]),
                    5
                ),
                suffix='reg'
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, .50, nan, nan, nan, nan, nan, 1.00]),
                func,
                (
                    array([5, 5, 5, 5, 5, 5, 1, 2, 3, 4, 5, 6]),
                    array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
                    5
                ),
                suffix='reg_category'
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, .50, nan, nan, nan, nan, nan, 1.00]),
                func,
                (
                    array([5, 5, 5, 5, 5, 5, 1, 2, 3, 4, 5, 6]),
                    array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
                    4
                ),
                suffix='error_period_less_5',
                exception=YaUberAlgoInternalError,
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, .50, nan, nan, nan, nan, nan, 1.00]),
                func,
                (
                    array([5, 5, 5, 5, 5, 5, 1, 2, 3, 4, 5, 6]),
                    array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
                    0
                ),
                suffix='error_period_zero',
                exception=YaUberAlgoArgumentError,
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, .50, nan, nan, nan, nan, nan, 1.00]),
                func,
                (
                    array([5, 5, 5, 5, 5, 5, 1, 2, 3, 4, 5, 6]),
                    array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
                    -1
                ),
                suffix='error_period_negative',
                exception=YaUberAlgoArgumentError,
            )

            s.check_regular(
                None,
                func,
                (
                    np.arange(0, 101),
                    np.arange(0, 101),
                    3,  # return_as_cat=
                ),
                suffix='category_more_than_100_unique_cats',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, .50, nan, nan, nan, nan, nan, nan, 1.00]),
                func,
                (
                    array([5, 5, 5, 5, 5, 5, 1, 2, 3, 4, 5, 6, 6]),
                    array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, nan, 0]),
                    5
                ),
                suffix='reg_category_nan_skipped'
            )

            rand_data = np.random.random(100)
            s.check_regular(
                percent_rank(rand_data, 5),
                func,
                (
                    rand_data,
                    np.ones(100),
                    5
                ),
                suffix='same_as_plain_rank'
            )

            rand_data = np.random.random(100)
            rand_data[25] = nan
            rand_data[49] = inf
            rand_data[59] = -inf
            s.check_naninf(
                percent_rank(rand_data, 5),
                func,
                (
                    rand_data,
                    np.ones(100),
                    5
                ),
                suffix='same_nan_inf_handling'
            )

            s.check_series(
                pd.Series(array([nan, nan, nan, nan, nan, .30, .10])),
                func,
                (
                    pd.Series(array([3, 2, 1, 4, 3, 2, 1])),
                    pd.Series(array([1, 1, 1, 1, 1, 1, 1])),
                    5
                ),
                suffix='reg'
            )

            s.check_dtype_float(
                array([nan, nan, nan, nan, nan, .30, .10], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1], dtype=np.float),
                    array([1, 1, 1, 1, 1, 1, 1], dtype=np.float),
                    5
                ),
                suffix='reg'
            )

            s.check_dtype_int(
                array([nan, nan, nan, nan, nan, .30, .10], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1], dtype=np.int32),
                    array([1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
                    5
                ),
                suffix='reg'
            )

            s.check_dtype_bool(
                array([nan, nan, nan, nan, nan, .20, .70], dtype=np.float),
                func,
                (
                    array([0, 1, 1, 0, 1, 0, 1], dtype=np.bool),
                    array([1, 1, 1, 1, 1, 1, 1], dtype=np.bool),
                    5
                ),
                suffix='reg'
            )

            s.check_dtype_object(
                func,
                (
                    array([0, 1, 1, 0, 1, 0, 1], dtype=np.object),
                    array([1, 1, 1, 1, 1, 1, 1], dtype=np.object),
                    5
                ),
                suffix='reg'
            )

            s.check_futref(5, 1,
                           func,
                           (
                               np.random.random(100),
                               np.random.random_integers(0, 3, 100),
                               5
                           ),
                           )

            s.check_window_consistency(5, 1,
                                       func,
                                       (
                                           np.random.random(100),
                                           np.random.random_integers(0, 3, 100),
                                           5
                                       ),
                                       )



