import unittest
from yauber_algo.errors import *


class RefTestCase(unittest.TestCase):
    def test_ref(self):
        import yauber_algo.sanitychecks as sc
        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import ref

        #
        # Function settings
        #
        algo = 'ref'
        func = ref

        setattr(sys.modules[func.__module__], 'IS_WARN_FUTREF', False)
        setattr(sys.modules[func.__module__], 'IS_RAISE_FUTREF', False)

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                array([3., 4., 5., 6., 7., 8., 9., 10., nan, nan]),
                func, (sc.SAMPLE_10_FLOAT, 2),
                suffix='fut_ref'
            )

            s.check_regular(
                array([3., 4., 5., 6., 7., 8., 9., 10., nan, nan]),
                func, ([2, 3, 4, 5, 6], 2),
                suffix='wrong_type',
                exception=YaUberAlgoArgumentError,
            )

            s.check_regular(
                pd.Series([nan, nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                          index=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),

                func, (np.array([True, nan], dtype=np.object), -2),
                suffix='wrong_type_ndarr',
                exception=YaUberAlgoDtypeNotSupportedError,
            )

            s.check_regular(
                pd.Series([nan, nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                          index=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),

                func, (pd.Series([True, nan], dtype=np.object), -2),
                suffix='wrong_type_series',
                exception=YaUberAlgoDtypeNotSupportedError,
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]),
                func, (sc.SAMPLE_10_FLOAT, -20),
                suffix='shift_too_low'
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]),
                func, (sc.SAMPLE_10_FLOAT, 20),
                suffix='shift_too_high'
            )

            s.check_regular(
                array([nan, nan, 1., 2., 3., 4., 5., 6., 7., 8.]),
                func, (sc.SAMPLE_10_FLOAT, -2),
                suffix='previous'
            )

            #
            # Check typing
            #
            s.check_series(
                pd.Series([nan, nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                          index=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),

                func, (sc.SAMPLE_10S_FLOAT, -2),
            )



            s.check_dtype_float(
                array([nan, nan, 1., 2., 3., 4., 5., 6., 7., 8.]),
                func, (sc.SAMPLE_10_FLOAT, -2),
            )

            s.check_dtype_bool(
                array([nan, 1., 1., 0., 0., 1., 1., 0., 0., 1.]),
                func, (sc.SAMPLE_10_BOOL, -1),
            )

            s.check_dtype_int(
                array([nan, nan, 1., 2., 3., 4., 5., 6., 7., 8.]),
                func, (sc.SAMPLE_10_INT, -2),
            )

            s.check_dtype_object(
                func, (sc.SAMPLE_10_OBJ, -2),
            )

            #
            # Check missing variables handling
            #
            s.check_naninf(
                array([nan, nan, 1., 2., nan, 4., 5., 6., nan, 8.]),
                func, (sc.SAMPLE_10_NANINF, -2),
                ignore_nan_argument_position_check=True
            )

            #
            # Check for future reference
            #
            s.check_futref(5, 1,
                           func, (sc.SAMPLE_10_FLOAT, -2),
                           min_checks_count=3,
                           )

            setattr(sys.modules[func.__module__], 'IS_WARN_FUTREF', True)
            setattr(sys.modules[func.__module__], 'IS_RAISE_FUTREF', True)
            s.check_futref(3, 1,
                           func, (sc.SAMPLE_10_FLOAT, 2),
                           expected=True,  # Warning: this means that we expect future reference
                           suffix='expected'
                           )

            setattr(sys.modules[func.__module__], 'IS_WARN_FUTREF', False)
            setattr(sys.modules[func.__module__], 'IS_RAISE_FUTREF', False)
            s.check_window_consistency(4, 1,
                                       func, (sc.SAMPLE_10_FLOAT, -3),
                                       # expected=True, # Warning: this means that we expect future reference
                                       # suffix='expected'
                                       )

            # s.check_window_consistency(4, 1,
            #    np.cumsum, (sc.SAMPLE_10_FLOAT,),
            #    #expected=True, # Warning: this means that we expect future reference
            #    suffix='expected'
            # )
