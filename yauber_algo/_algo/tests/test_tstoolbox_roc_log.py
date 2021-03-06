import unittest
from yauber_algo.errors import *


class RocLogTestCase(unittest.TestCase):
    def test_roc_log(self):
        import yauber_algo.sanitychecks as sc
        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np
        from math import log

        from yauber_algo.algo import roc_log

        #
        # Function settings
        #
        algo = 'roc_log'
        func = roc_log

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                array([nan, 0, 0, 0, 0, 0, 0]),
                func,
                (
                    array([3, 3, 3, 3, 3, 3, 3]),
                    1
                ),
                suffix='no_change'
            )

            s.check_regular(
                array([nan, log(4/3), log(3/4), 0, 0, 0, 0]),
                func,
                (
                    array([3, 4, 3, 3, 3, 3, 3]),
                    1
                ),
                suffix='1period'
            )

            s.check_regular(
                array([nan, nan, log(4/3), 0, log(3/4), 0, 0]),
                func,
                (
                    array([3, 3, 4, 3, 3, 3, 3]),
                    2
                ),
                suffix='2period'
            )

            s.check_regular(
                array([nan, 0, 0, 0, 0, 0, 0]),
                func,
                (
                    array([0, 3, 3, 3, 3, 3, 3]),
                    1
                ),
                suffix='zeros_not_allowed',
                exception=YaUberAlgoInternalError
            )

            s.check_regular(
                array([nan, 0, 0, 0, 0, 0, 0]),
                func,
                (
                    array([3, 3, 3, 3, 3, 3, 0]),
                    1
                ),
                suffix='zeros_not_allowed_at_end',
                exception=YaUberAlgoInternalError
            )

            s.check_regular(
                array([nan, 0, 0, 0, 0, 0, 0]),
                func,
                (
                    array([-3, 3, 3, 3, 3, 3, 3]),
                    1
                ),
                suffix='negative_not_allowed',
                exception=YaUberAlgoInternalError
            )

            s.check_regular(
                array([nan, 0, 0, 0, 0, 0, 0]),
                func,
                (
                    array([3, 3, 3, 3, 3, 3, -3]),
                    1
                ),
                suffix='negative_not_allowed_at_end',
                exception=YaUberAlgoInternalError
            )

            s.check_regular(
                array([nan, 0, 0, 0, 0, 0, 0]),
                func,
                (
                    array([3, 3, 3, 3, 3, 3, 3]),
                    0
                ),
                suffix='no_zero_period',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([nan, 0, 0, 0, 0, 0, 0]),
                func,
                (
                    array([3, 3, 3, 3, 3, 3, 3]),
                    -1
                ),
                suffix='no_negative_period',
                exception=YaUberAlgoArgumentError
            )

            s.check_series(
                pd.Series(array([nan, log(4 / 3), log(3 / 4), 0, 0, 0, 0])),
                func,
                (
                    pd.Series(array([3, 4, 3, 3, 3, 3, 3])),
                    1
                ),
                suffix='1period'
            )

            s.check_naninf(
                array([nan, nan, log(3 / 4), 0, 0, nan, nan]),
                func,
                (
                    array([nan, 4, 3, 3, 3, inf, nan]),
                    1
                ),
                suffix='1period'
            )

            s.check_dtype_float(
                array([nan, log(4 / 3), log(3 / 4), 0, 0, 0, 0], dtype=np.float),
                func,
                (
                    array([3, 4, 3, 3, 3, 3, 3], dtype=np.float),
                    1
                ),
                suffix='1period'
            )

            s.check_dtype_int(
                array([nan, log(4 / 3), log(3 / 4), 0, 0, 0, 0], dtype=np.float),
                func,
                (
                    array([3, 4, 3, 3, 3, 3, 3], dtype=np.int32),
                    1
                ),
                suffix='1period'
            )

            s.check_dtype_bool(
                array([nan, 0, 0, 0, 0, 0, 0], dtype=np.float),
                func,
                (
                    array([1, 1, 1, 1, 1, 1, 1], dtype=np.bool),
                    1
                ),
                suffix='1period',
                exception=YaUberAlgoDtypeNotSupportedError
            )

            s.check_dtype_object(
                func,
                (
                    array([3, 4, 3, 3, 3, 3, 3], dtype=np.object),
                    1
                ),
                suffix='1period'
            )

            s.check_futref(5, 1,
                           func,
                           (
                               # This test might fail due to YaUberInternalError - because of zero values
                               np.random.random(100),
                               5
                           ),
                           )

            s.check_window_consistency(5, 1,
                                       func,
                                       (
                                           # This test might fail due to YaUberInternalError - because of zero values
                                           np.random.random(100),
                                           5
                                       ),
                                       )
