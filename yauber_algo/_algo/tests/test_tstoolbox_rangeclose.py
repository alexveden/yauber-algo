import unittest
from yauber_algo.errors import *


class RangeCloseTestCase(unittest.TestCase):
    def test_rangeclose(self):
        import yauber_algo.sanitychecks as sc
        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import rangeclose
        import yauber_algo.algo as a

        #
        # Function settings
        #
        algo = 'rangeclose'
        func = rangeclose

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #


            s.check_regular(
                array([0.8]),
                func,
                (
                    array([10]),  # h
                    array([ 0]),  # l
                    array([ 8]),  # c
                    1
                ),
                suffix='uptrend'
            )

            s.check_regular(
                array([0.8]),
                func,
                (
                    array([10]),  # h
                    array([0]),  # l
                    array([8]),  # c
                    0
                ),
                suffix='period_zero',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([0.4]),
                func,
                (
                    array([10]),  # h
                    array([0]),  # l
                    array([8]),  # c
                    -1
                ),
                suffix='period_negative',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([0.4]),
                func,
                (
                    array([10]),  # h
                    array([0]),  # l
                    array([11]),  # c
                    1
                ),
                suffix='close_gt_hi',
                exception=YaUberAlgoInternalError
            )

            s.check_regular(
                array([0.4]),
                func,
                (
                    array([10]),  # h
                    array([0]),  # l
                    array([-1]),  # c
                    1
                ),
                suffix='close_lt_lo',
                exception=YaUberAlgoInternalError
            )

            s.check_regular(
                array([0.2]),
                func,
                (
                    array([10]),  # h
                    array([0]),   # l
                    array([2]),   # c
                    1
                ),
                suffix='dn_trend'
            )

            s.check_regular(
                array([0.5]),
                func,
                (
                    array([10]),  # h
                    array([0]),  # l
                    array([5]),  # c
                    1
                ),
                suffix='neutral_trend'
            )

            s.check_regular(
                array([0.4]),
                func,
                (
                    array([0]),  # h
                    array([10]),  # l
                    array([2]),  # c
                    1
                ),
                suffix='sanity_h_less_l',
                exception=YaUberAlgoInternalError
            )

            s.check_regular(
                array([0.5]),
                func,
                (
                    array([10]),  # h
                    array([0]),  # l
                    array([5]),  # c
                    1
                ),
                suffix='doji'
            )

            s.check_regular(
                array([0.5]),
                func,
                (
                    array([10]),  # h
                    array([0]),  # l
                    array([5]),  # c
                    1
                ),
                suffix='doji_h_eq_l'
            )

            s.check_regular(
                array([1.0]),
                func,
                (
                    array([10]),  # h
                    array([0]),  # l
                    array([10]),  # c
                    1
                ),
                suffix='ideal_trend_up'
            )

            s.check_regular(
                array([0.0]),
                func,
                (
                    array([10]),  # h
                    array([0]),  # l
                    array([0]),  # c
                    1
                ),
                suffix='ideal_trend_dn'
            )

            s.check_regular(
                array([0.5]),
                func,
                (
                    array([10]),  # h
                    array([10]),  # l
                    array([10]),  # c
                    1
                ),
                suffix='flat_candle'
            )


            s.check_regular(
                array([nan]),
                func,
                (
                    array([nan]),  # h
                    array([10]),  # l
                    array([10]),  # c
                    1
                ),
                suffix='nan_h'
            )

            s.check_regular(
                array([nan]),
                func,
                (
                    array([10]),  # h
                    array([nan]),  # l
                    array([10]),  # c
                    1
                ),
                suffix='nan_l'
            )

            s.check_regular(
                array([nan]),
                func,
                (
                    array([10]),  # h
                    array([10]),  # l
                    array([nan]),  # c
                    1
                ),
                suffix='nan_c'
            )

            s.check_regular(
                array([nan, 0.5]),
                func,
                (
                    array([6, 10]),  # h
                    array([0, 3]),  # l
                    array([0, 5]),  # c
                    2
                ),
                suffix='period2'
            )


            s.check_regular(
                array([nan, 1.0]),
                func,
                (
                    array([10, 10]),  # h
                    array([0, 3]),  # l
                    array([0, 10]),  # c
                    2
                ),
                suffix='period2_alt'
            )

            s.check_regular(
                array([nan, 1.0, (7-0.0)/(20-0)]),
                func,
                (
                    array([10, 10, 20]),  # h
                    array([0, 3, 0]),  # l
                    array([0, 10, 7]),  # c
                    2
                ),
                suffix='period2_alt_n3'
            )

            s.check_regular(
                array([nan, nan, (7-0)/(20-0)]),
                func,
                (
                    array([10, 10, 20]),  # h
                    array([0, nan, 0]),  # l
                    array([0, 10, 7]),  # c
                    2
                ),
                suffix='period2_alt_n3_llv_nan'
            )

            s.check_regular(
                array([nan, (5-0)/(10-0), (2-0) / (5-0)]),
                func,
                (
                    array([10, 5, 2]),  # h
                    array([0, 3, 0]),  # l
                    array([0, 5, 2]),  # c
                    2
                ),
                suffix='period2_alt_n3_hhv'
            )

            s.check_regular(
                array([nan, nan, 0.7]),
                func,
                (
                    array([10, nan, 10]),  # h
                    array([0, 3, 0]),  # l
                    array([0, 10, 7]),  # c
                    2
                ),
                suffix='period2_alt_n3_hhv_with_nan'
            )

            s.check_naninf(
                array([nan, 0.5]),
                func,
                (
                    array([nan, 10]),  # h
                    array([inf, 0]),  # l
                    array([nan, 5]),  # c
                    2
                ),
                suffix='period2_nan_ignored',
            )

            s.check_series(
                pd.Series(array([nan, 0.5])),
                func,
                (
                    pd.Series(array([6, 10])),  # h
                    pd.Series(array([0, 3])),  # l
                    pd.Series(array([0, 5])),  # c
                    2
                ),
                suffix='series'
            )

            s.check_dtype_float(
                array([0.8], dtype=np.float),
                func,
                (
                    array([10], dtype=np.float),  # h
                    array([0], dtype=np.float),  # l
                    array([8], dtype=np.float),  # c
                    1
                ),
            )

            s.check_dtype_int(
                array([0.8], dtype=np.float),
                func,
                (
                    array([10], dtype=np.int32),  # h
                    array([0], dtype=np.int32),  # l
                    array([8], dtype=np.int32),  # c
                    1
                ),
            )

            s.check_dtype_bool(
                array([0.8], dtype=np.float),
                func,
                (
                    array([2], dtype=np.bool),  # o
                    array([0], dtype=np.int32),  # l
                    array([8], dtype=np.int32),  # c
                    1
                ),
                suffix='bool_h',
                exception=YaUberAlgoDtypeNotSupportedError
            )

            s.check_dtype_bool(
                array([0.4], dtype=np.float),
                func,
                (
                    array([0], dtype=np.int32),  # l
                    array([2], dtype=np.bool),  # o
                    array([8], dtype=np.int32),  # c
                    1
                ),
                suffix='bool_l',
                exception=YaUberAlgoDtypeNotSupportedError
            )

            s.check_dtype_bool(
                array([0.4], dtype=np.float),
                func,
                (
                    array([0], dtype=np.int32),  # l
                    array([8], dtype=np.int32),  # c
                    array([2], dtype=np.bool),  # o
                    1
                ),
                suffix='bool_c',
                exception=YaUberAlgoDtypeNotSupportedError
            )

            s.check_dtype_object(
                func,
                (
                    array([2], dtype=np.object),  # o
                    array([0], dtype=np.int32),  # l
                    array([8], dtype=np.int32),  # c
                    1
                ),
                suffix='bool_o',
            )



            # Check with manual formula
            h = l = c = o = np.random.random(1000)
            rclose = (c - a.llv(l, 20)) / (a.hhv(h, 20) - a.llv(l, 20))

            s.check_regular(
                rclose,
                func,
                (
                    h,
                    l,
                    c,
                    20
                ),
                suffix='formula'
            )

            s.check_futref(5, 1,
                           func,
                           (
                               np.random.random_integers(60, 100, 100),  # h
                               np.random.random_integers(0,  30, 100),  # l
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


