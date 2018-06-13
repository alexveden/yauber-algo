import unittest
from yauber_algo.errors import *


class ApplyTestCase(unittest.TestCase):
    def test_percent_rank_category(self):
        import yauber_algo.sanitychecks as sc
        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import apply

        #
        # Function settings
        #
        algo = 'apply'
        func = apply

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                array([nan, nan, 6, 7, 8, 9, 6]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    3,
                    np.sum,
                    None,  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='rolling'
            )

            s.check_regular(
                array([nan, nan, 6, nan, nan, 9, 6]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    3,
                    np.sum,
                    array([0, 0, 0, 1, 1, 1, 1]),  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='category'
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, nan, nan]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    3,
                    np.sum,
                    array([0, 0, 0, 1, 1, 1, 1]),  # category=
                    3,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='category_ret_as_cat_number_not_exists'
            )

            s.check_regular(
                None,
                func,
                (
                    np.arange(0, 101),
                    3,
                    np.sum,
                    np.arange(0, 101),
                    3,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='category_more_than_100_unique_cats',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([nan, nan, 6, 6, 6, 6, 6]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    3,
                    np.sum,
                    array([0, 0, 0, 1, 1, 1, 1]),  # category=
                    0,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='category_exact'
            )

            s.check_regular(
                array([nan, nan, 6, 6, 6, 6, 6]),
                func,
                (
                    array([3, 2, 1, nan, nan, nan, nan]),
                    3,
                    np.sum,
                    array([0, 0, 0, 1, 1, 1, 1]),  # category=
                    0,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='category_ret_nan'
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, nan, nan]),
                func,
                (
                    array([3, 2, 1, 4, 1, nan, nan]),
                    3,
                    np.sum,
                    array([0, 0, 0, 1, 1, 1, 1]),  # category=
                    array([1, 1, 1, 1, 1, 1, 1]),  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='category_ret_nan_if_arr_nan'
            )




            s.check_regular(
                array([nan, nan, 6, 7, 8, 9, 6]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    0,
                    np.sum,
                    None,  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='zero_period',
                exception=YaUberAlgoArgumentError,
            )

            s.check_regular(
                array([nan, nan, 6, 7, 8, 9, 6]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    -1,
                    np.sum,
                    None,  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='neg_period',
                exception=YaUberAlgoArgumentError,
            )

            s.check_regular(
                func(array([3, 2, 1, 4, 3, 2, 1]), 3, np.sum),

                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    3,
                    np.sum,
                    array([1, 1, 1, 1, 1, 1, 1]),  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='rolling_and_categorical_equal'
            )

            s.check_regular(
                array([nan, nan, nan, nan, 8, 9, 6]),
                func,
                (
                    array([3, nan, 1, 4, 3, 2, 1]),
                    3,
                    np.sum,
                    None,  # category=
                    None,  # return_as_cat=
                    False,  # exclude_nan=
                ),
                suffix='rolling_not_exclude_nan'
            )
            #
            #  NAN / INF
            #
            #

            s.check_naninf(
                array([nan, nan, nan, nan, 8, 9, nan]),
                func,
                (
                    array([inf, nan, 1, 4, 3, 2, inf]),
                    3,
                    np.sum,
                    None,  # category=
                    None,  # return_as_cat=
                    False,  # exclude_nan=
                ),
                suffix='rolling'
            )

            s.check_naninf(
                array([nan, nan, 1, 5, 8, 9, nan]),
                func,
                (
                    array([inf, nan, 1, 4, 3, 2, inf]),
                    3,
                    np.sum,
                    None,  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='rolling_naninf_excluded'
            )


            s.check_series(
                pd.Series(array([nan, nan, 6, 7, 8, 9, 6])),
                func,
                (
                    pd.Series(array([3, 2, 1, 4, 3, 2, 1])),
                    3,
                    np.sum,
                    None,  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='rolling'
            )

            s.check_series(
                pd.Series(array([nan, nan, 6, 7, 8, 9, 6])),
                func,
                (
                    pd.Series(array([3, 2, 1, 4, 3, 2, 1])),
                    3,
                    np.sum,
                    pd.Series(array([0, 0, 0, 0, 0, 0, 0])),  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='categorical'
            )

            s.check_series(
                pd.Series(array([nan, nan, 6, 7, 8, 9, 6])),
                func,
                (
                    pd.Series(array([3, 2, 1, 4, 3, 2, 1])),
                    3,
                    np.sum,
                    pd.Series(array([0, 0, 0, 0, 0, 0, 0])),  # category=
                    pd.Series(array([0, 0, 0, 0, 0, 0, 0])),  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='categorical_ret_as'
            )

            s.check_regular(
                array([nan, nan, 6, 7, nan, nan, 6]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    3,
                    np.sum,
                    array([0, 0, 0, 0, 1, 1, 1]),  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='categorical'
            )

            s.check_naninf(
                array([nan, nan, 6, nan, nan, nan, nan]),
                func,
                (
                    array([3, 2, 1, nan, 3, 2, inf]),
                    3,
                    np.sum,
                    array([0, 0, 0, 0, 1, 1, 1]),  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='categorical'
            )

            s.check_naninf(
                array([nan, nan, 6, nan, nan, nan, nan]),
                func,
                (
                    array([3, 2, 1, 2, 3, 2, 4]),
                    3,
                    np.sum,
                    array([0, 0, 0, inf, 1, 1, nan]),  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='nan_for_category'
            )

            s.check_naninf(
                array([nan, nan, 6, 6, 6, 6, 6]),
                func,
                (
                    array([3, 2, 1, 2, 3, 2, 4]),
                    3,
                    np.sum,
                    array([0, 0, 0, inf, 1, 1, nan]),  # category=
                    0,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='return_as_cat_ignore_codex',
                ignore_nan_argument_position_check=True,
            )

            s.check_naninf(
                array([nan, nan, nan, nan, nan, 6, 6]),
                func,
                (
                    array([3, 2, 1, 2, 3, 2, nan]),
                    3,
                    np.sum,
                    array([0, 0, 1, inf, 1, 1, nan]),  # category=
                    1,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='return_as_cat_non_NAN_if_reference_with_valid_window',
                ignore_nan_argument_position_check=True,
            )

            s.check_naninf(
                array([nan, nan, nan, nan, nan, 6, nan]),
                func,
                (
                    array([3, 2, 1, 2, 3, 2, nan]),
                    3,
                    np.sum,
                    array([0, 0, 1, inf, 1, 1, 1]),  # category=
                    1,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='return_as_cat_NOT_ignore_codex_if_same_cat',
            )

            s.check_naninf(
                array([nan, nan, nan, nan, nan, nan, nan]),
                func,
                (
                    array([3, 2, 1, 2, 3, 2, nan]),
                    3,
                    np.sum,
                    array([0, 0, 1, inf, 1, 1, nan]),  # category=
                    0,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='return_as_cat_widows_less_period',
            )


            s.check_dtype_float(
                array([nan, nan, 6, 7, 8, 9, 6], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1], dtype=np.float),
                    3,
                    np.sum,
                    None,  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='rolling'
            )

            s.check_dtype_float(
                array([nan, nan, 6, 5, nan, nan, 9], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 2, 3, 2, 4], dtype=np.float),
                    3,
                    np.sum,
                    array([0, 0, 0, 0, 1, 1, 1], dtype=np.float),  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='category'
            )

            s.check_dtype_float(
                array([nan, nan, 6, 5, 5, 5, 5], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 2, 3, 2, 4], dtype=np.float),
                    3,
                    np.sum,
                    array([0, 0, 0, 0, 1, 1, 1], dtype=np.float),  # category=
                    array([0, 0, 0, 0, 0, 0, 0], dtype=np.float),
                    True,  # exclude_nan=
                ),
                suffix='category_ret_as'
            )

            s.check_dtype_bool(
                array([nan, nan, 3, 3, 3, 3, 3], dtype=np.float),
                func,
                (
                    array([1, 1, 1, 1, 1, 1, 1], dtype=np.bool),
                    3,
                    np.sum,
                    None,  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='rolling'
            )

            s.check_dtype_bool(
                array([nan, nan, 3, 3, nan, nan, 3], dtype=np.float),
                func,
                (
                    array([1, 1, 1, 1, 1, 1, 1], dtype=np.bool),
                    3,
                    np.sum,
                    array([0, 0, 0, 0, 1, 1, 1], dtype=np.bool),  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='category'
            )

            s.check_dtype_bool(
                array([nan, nan, 6, 5, 5, 5, 5], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 2, 3, 2, 4], dtype=np.float),
                    3,
                    np.sum,
                    array([0, 0, 0, 0, 1, 1, 1], dtype=np.bool),  # category=
                    array([0, 0, 0, 0, 0, 0, 0], dtype=np.bool),
                    True,  # exclude_nan=
                ),
                suffix='category_ret_as'
            )

            s.check_dtype_int(
                array([nan, nan, 6, 7, 8, 9, 6], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1], dtype=np.int32),
                    3,
                    np.sum,
                    None,  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='rolling'
            )

            s.check_dtype_int(
                array([nan, nan, 6, 5, nan, nan, 9], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 2, 3, 2, 4], dtype=np.int32),
                    3,
                    np.sum,
                    array([0, 0, 0, 0, 1, 1, 1], dtype=np.int32),  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='category'
            )

            s.check_dtype_int(
                array([nan, nan, 6, 5, 5, 5, 5], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 2, 3, 2, 4], dtype=np.float),
                    3,
                    np.sum,
                    array([0, 0, 0, 0, 1, 1, 1], dtype=np.int32),  # category=
                    array([0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
                    True,  # exclude_nan=
                ),
                suffix='category_ret_as'
            )

            s.check_dtype_object(
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1], dtype=np.object),
                    3,
                    np.sum,
                    None,  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='rolling'
            )

            s.check_dtype_object(
                func,
                (
                    array([3, 2, 1, 2, 3, 2, 4], dtype=np.object),
                    3,
                    np.sum,
                    array([0, 0, 0, 0, 1, 1, 1], dtype=np.object),  # category=
                    None,  # return_as_cat=
                    True,  # exclude_nan=
                ),
                suffix='category'
            )

            s.check_dtype_object(
                func,
                (
                    array([3, 2, 1, 2, 3, 2, 4], dtype=np.float),
                    3,
                    np.sum,
                    array([0, 0, 0, 0, 1, 1, 1], dtype=np.float),  # category=
                    array([0, 0, 0, 0, 0, 0, 0], dtype=np.object),
                    True,  # exclude_nan=
                ),
                suffix='category_ret_as'
            )

            s.check_futref(5, 1,
                           func,
                           (
                               np.random.random(100),
                               5,
                               np.sum,
                           ),
                           suffix='rolling'
                           )

            s.check_window_consistency(5, 1,
                                       func,
                                       (
                                           np.random.random(100),
                                           5,
                                           np.sum,
                                       ),
                                       suffix='rolling'
                                       )

            s.check_futref(5, 1,
                           func,
                           (
                               np.random.random(100),
                               5,
                               np.sum,
                               np.random.random_integers(0, 3, 100),
                           ),
                           suffix='category'
                           )

            s.check_window_consistency(5, 1,
                                       func,
                                       (
                                           np.random.random(100),
                                           5,
                                           np.sum,
                                           np.random.random_integers(0, 3, 100),
                                       ),
                                       suffix='category'
                                       )

            s.check_futref(5, 1,
                           func,
                           (
                               np.random.random(100),
                               5,
                               np.sum,
                               np.random.random_integers(0, 3, 100),
                               np.random.random_integers(0, 3, 100),
                           ),
                           suffix='category_ret_as'
                           )

            s.check_window_consistency(5, 1,
                                       func,
                                       (
                                           np.random.random(100),
                                           5,
                                           np.sum,
                                           np.random.random_integers(0, 3, 100),
                                           np.random.random_integers(0, 3, 100),
                                       ),
                                       suffix='category_ret_as'
                                       )



