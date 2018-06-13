import unittest
from yauber_algo.errors import *


class TrueRangeTestCase(unittest.TestCase):
    def test_truerange(self):
        import yauber_algo.sanitychecks as sc
        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import truerange
        import yauber_algo.algo as a

        #
        # Function settings
        #
        algo = 'truerange'
        func = truerange

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #


            s.check_regular(
                array([nan]),
                func,
                (
                    array([10]),  # h
                    array([0]),  # l
                    array([8]),  # c
                    1
                ),
                suffix='first_is_nan'
            )

            s.check_regular(
                array([nan, 10]),
                func,
                (
                    array([10, 10]),  # h
                    array([0, 0]),  # l
                    array([8, 8]),  # c
                    1
                ),
                suffix='hl_is_greater'
            )

            s.check_regular(
                array([nan, 72]),
                func,
                (
                    array([10, 80]),  # h
                    array([0, 75]),  # l
                    array([8, 80]),  # c
                    1
                ),
                suffix='c-c_is_greater'
            )

            s.check_regular(
                array([nan, 72]),
                func,
                (
                    array([80, 20]),  # h
                    array([0, 8]),  # l
                    array([80, 8]),  # c
                    1
                ),
                suffix='c-c_is_greater_abs'
            )

            s.check_regular(
                array([nan, nan, 80-8]),
                func,
                (
                    array([80, 20, 20]),  # h
                    array([0,  8, 8]),  # l
                    array([80, 8, 8]),  # c
                    2
                ),
                suffix='2_period_c_isgreater'
            )

            s.check_regular(
                array([nan, nan, nan]),
                func,
                (
                    array([8, 8, 8]),  # h
                    array([8, 8, 8]),  # l
                    array([8, 8, 8]),  # c
                    2
                ),
                suffix='all_same'
            )

            s.check_regular(
                array([nan, nan, 50 - 8]),
                func,
                (
                    array([80, 20, 50]),  # h
                    array([0, 8, 8]),  # l
                    array([8, 8, 8]),  # c
                    2
                ),
                suffix='2_period_hl'
            )

            s.check_regular(
                array([nan, nan, nan]),
                func,
                (
                    array([80, 20, nan]),  # h
                    array([0, 8, 8]),  # l
                    array([8, 8, 8]),  # c
                    2
                ),
                suffix='nan_h'
            )

            s.check_regular(
                array([nan, nan, 20 - 8]),
                func,
                (
                    array([80, nan, 20]),  # h
                    array([0, 8, 8]),  # l
                    array([8, 8, 8]),  # c
                    2
                ),
                suffix='nan_h_skip'
            )

            s.check_regular(
                array([nan, nan, nan]),
                func,
                (
                    array([80, 20, 50]),  # h
                    array([0, 8, nan]),  # l
                    array([8, 8, 8]),  # c
                    2
                ),
                suffix='nan_l'
            )

            s.check_regular(
                array([nan, nan, 50 - 8]),
                func,
                (
                    array([80, 50, 20]),  # h
                    array([0, nan, 8]),  # l
                    array([8, 8, 8]),  # c
                    2
                ),
                suffix='nan_l_skip'
            )

            s.check_regular(
                array([nan, nan, nan]),
                func,
                (
                    array([80, 20, 20]),  # h
                    array([0, 8, 8]),  # l
                    array([nan, 8, 8]),  # c
                    2
                ),
                suffix='2_period_c_isgreater_nan_skip'
            )

            s.check_regular(
                array([nan, nan, nan]),
                func,
                (
                    array([80, 20, 50]),  # h
                    array([0, 8, 0]),  # l
                    array([8, 8, nan]),  # c
                    2
                ),
                suffix='nan_c'
            )

            s.check_regular(
                array([nan, nan, 50 - 8]),
                func,
                (
                    array([80, 50, 20]),  # h
                    array([0, 8, 8]),  # l
                    array([8, nan, 8]),  # c
                    2
                ),
                suffix='nan_c_skip'
            )

            s.check_regular(
                array([nan, nan, 50 - 8]),
                func,
                (
                    array([80, 50, 20]),  # h
                    array([0, 8, 8]),  # l
                    array([8, 8, 100]),  # c
                    2
                ),
                suffix='nan_c_gt_h',
                exception=YaUberAlgoInternalError
            )

            s.check_regular(
                array([nan, nan, 50 - 8]),
                func,
                (
                    array([80, 50, 20]),  # h
                    array([0, 8, 8]),  # l
                    array([8, 8, 100]),  # c
                    0
                ),
                suffix='period_must_be_positive',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([nan, nan, 50 - 8]),
                func,
                (
                    array([80, 50, 20]),  # h
                    array([0, 8, 8]),  # l
                    array([8, 8, 7]),  # c
                    2
                ),
                suffix='nan_c_lt_l',
                exception=YaUberAlgoInternalError
            )

            s.check_regular(
                array([nan, nan, 50 - 8]),
                func,
                (
                    array([80, 50, 7]),  # h
                    array([0, 8, 8]),  # l
                    array([8, 8, 8]),  # c
                    2
                ),
                suffix='nan_h_lt_l',
                exception=YaUberAlgoInternalError
            )

            s.check_naninf(
                array([nan, nan, nan]),
                func,
                (
                    array([nan, 20, inf]),  # h
                    array([0, 8, 8]),  # l
                    array([0, 8, 8]),  # c
                    2
                ),
                suffix='h_inf'
            )

            s.check_naninf(
                array([nan, nan, nan]),
                func,
                (
                    array([0, 8, 8]),  # l
                    array([nan, 8, inf]),  # h
                    array([8, 8, 8]),  # c
                    2
                ),
                suffix='l_inf'
            )

            s.check_naninf(
                array([nan, nan, nan]),
                func,
                (
                    array([0, 8, 8]),  # l
                    array([8, 8, 8]),  # c
                    array([nan, 8, inf]),  # h

                    2
                ),
                suffix='c_inf'
            )

            s.check_series(
                pd.Series(array([nan, 10])),
                func,
                (
                    pd.Series(array([10, 10])),  # h
                    pd.Series(array([0, 0])),  # l
                    pd.Series(array([8, 8])),  # c
                    1
                ),
                suffix=''
            )

            s.check_dtype_float(
                array([nan, 10], dtype=np.float),
                func,
                (
                    array([10, 10], dtype=np.float),  # h
                    array([0, 0], dtype=np.float),  # l
                    array([8, 8], dtype=np.float),  # c
                    1
                ),
                suffix=''
            )

            s.check_dtype_bool(
                array([nan, 10], dtype=np.float),
                func,
                (
                    array([10, 10], dtype=np.bool),  # h
                    array([0, 0], dtype=np.float),  # l
                    array([8, 8], dtype=np.float),  # c
                    1
                ),
                suffix='',
                exception=YaUberAlgoDtypeNotSupportedError
            )

            s.check_dtype_int(
                array([nan, 10], dtype=np.float),
                func,
                (
                    array([10, 10], dtype=np.int32),  # h
                    array([0, 0], dtype=np.int32),  # l
                    array([8, 8], dtype=np.int32),  # c
                    1
                ),
                suffix=''
            )

            s.check_dtype_object(
                func,
                (
                    array([10, 10], dtype=np.object),  # h
                    array([0, 0], dtype=np.int32),  # l
                    array([8, 8], dtype=np.int32),  # c
                    1
                ),
                suffix=''
            )

            s.check_futref(5, 1,
                           func,
                           (
                               np.random.random_integers(60, 100, 100),  # h
                               np.random.random_integers(0, 30, 100),  # l
                               np.random.random_integers(31, 59, 100),  # c
                               5
                           ),
                           )

            s.check_window_consistency(5, 1,
                                       func,
                                       (
                                           np.random.random_integers(60, 100, 100),  # h
                                           np.random.random_integers(0, 30, 100),  # l
                                           np.random.random_integers(31, 59, 100),  # c
                                           5
                                       ),
                                       )




