import unittest
from ..algo import ref
import yauber_algo.sanitychecks as sc
import pandas as pd
import numpy as np
from yauber_algo.errors import YaUberSanityCheckError, YaUberAlgoFutureReferenceError, YaUberAlgoWindowConsistencyError, YaUberAlgoDtypeNotSupportedError
from io import StringIO
from unittest.mock import patch, MagicMock
import sys
import warnings


class SanityChecksTestCase(unittest.TestCase):
    def test_test_names(self):
        self.assertEqual(sc.TESTS_NAMES, ['regular', 'series', 'naninf', 'dtype_float',
                                          'dtype_bool', 'dtype_int', 'dtype_object', 'futref', 'window_consistency'])

    def test_samples(self):
        self.assertTrue(np.all(sc.SAMPLE_10_FLOAT == np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float)))
        self.assertTrue(np.allclose(sc.SAMPLE_10_NANINF, np.array([1, 2, np.nan, 4, 5, 6, np.inf, 8, 9, 10], dtype=np.float), equal_nan=True))

        self.assertTrue(np.allclose(sc.SAMPLE_10_BOOL, np.array([True, True, False, False, True, True, False, False, True, True], dtype=np.bool)))

        self.assertTrue(np.allclose(sc.SAMPLE_10_INT, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)))
        for a, b in zip(sc.SAMPLE_10_OBJ,
                        np.array([True, True, False, False, True, np.nan, False, False, True, True], dtype=np.object)):
            if a != b:
                if isinstance(a, float) and isinstance(b, float):
                    self.assertTrue(np.isfinite(a) == np.isfinite(b))
                    if np.isfinite(a):
                        self.assertEqual(a, b)
                else:
                    self.assertEqual(a, b)

        self.assertTrue(np.allclose(sc.SAMPLE_10S_FLOAT, pd.Series(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                                                   index=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
                                                                   dtype=np.float)
                                    ))

    def test_init(self):
        s = sc.SanityChecker('algo_test', debug=True)

        self.assertEqual(s.tests, {})
        self.assertEqual(s.tests_suppressed, set())
        self.assertEqual(s.debug, True)

    def test_supress_test(self):
        s = sc.SanityChecker('algo_test', debug=True)
        self.assertEqual(s.tests_suppressed, set())

        s.suppress_test('futref')
        self.assertEqual(True, 'futref' in s.tests_suppressed)

        self.assertRaises(ValueError, s.suppress_test, 'some_unknown_test_name')


    def test_enter_exit(self):
        try:
            with sc.SanityChecker('algo_test', debug=True) as s:
                self.assertEqual(True, isinstance(s, sc.SanityChecker))
                self.assertEqual(s.tests, {test_name: None for test_name in sc.TESTS_NAMES})

                s.tests['regular'] = {'base': True}
                s.tests['naninf'] = {'': False}
                s.tests_suppressed.add('dtype_object')

            self.assertTrue(False, 'sc.SanityChecker must throw YaUberSanityCheckError on __exit__')
        except YaUberSanityCheckError:
            pass

        # Just for debug line
        with sc.SanityChecker('algo_test', debug=True) as s:
            self.assertEqual(True, isinstance(s, sc.SanityChecker))
            self.assertEqual(s.tests, {test_name: None for test_name in sc.TESTS_NAMES})
            for t_name in sc.TESTS_NAMES:
                s.tests_suppressed.add(t_name)

    def test_series_compare_ndarray(self):
        self.assertRaises(AssertionError, sc.SanityChecker.series_compare,
                          np.array([1, 2, 3]),
                          pd.Series([1, 2, 3]))

        self.assertRaises(AssertionError, sc.SanityChecker.series_compare,
                          np.array([1, 2, 3], dtype=np.float),
                          np.array([1, 2], dtype=np.float))

        self.assertRaises(AssertionError, sc.SanityChecker.series_compare,
                          np.array([1, 2, 3], dtype=np.int32),
                          np.array([1, 2, 3], dtype=np.float))

        self.assertRaises(AssertionError, sc.SanityChecker.series_compare,
                          np.array([1, 2, 3], dtype=np.float),
                          np.array([1, 2, 3], dtype=np.int32))

        self.assertRaises(AssertionError, sc.SanityChecker.series_compare,
                          np.array([1, 2, np.nan], dtype=np.float),
                          np.array([1, 1, np.nan], dtype=np.float))

        sc.SanityChecker.series_compare(np.array([1, 2, np.nan], dtype=np.float),
                                        np.array([1, 2, np.nan], dtype=np.float))

        self.assertRaises(AssertionError, sc.SanityChecker.series_compare,
                          np.array([1, 2, np.inf], dtype=np.float),
                          np.array([1, 2, np.inf], dtype=np.float))

    def test_series_compare_pd_series(self):

        self.assertRaises(AssertionError, sc.SanityChecker.series_compare,
                          pd.Series([1, 2, 3], dtype=np.int32),
                          pd.Series([1, 2, 3], dtype=np.float))

        self.assertRaises(AssertionError, sc.SanityChecker.series_compare,
                          pd.Series([1, 2, 3], dtype=np.float),
                          pd.Series([1, 2, 3], dtype=np.int32))

        self.assertRaises(AssertionError, sc.SanityChecker.series_compare,
                          pd.Series([1, 2, np.nan], dtype=np.float),
                          pd.Series([1, 1, np.nan], dtype=np.float))

        sc.SanityChecker.series_compare(pd.Series([1, 2, np.nan], dtype=np.float),
                                        pd.Series([1, 2, np.nan], dtype=np.float))

        self.assertRaises(AssertionError, sc.SanityChecker.series_compare,
                          pd.Series([1, 2, np.inf], dtype=np.float),
                          pd.Series([1, 2, np.inf], dtype=np.float))

        self.assertRaises(AssertionError, sc.SanityChecker.series_compare,
                          pd.Series([1, 2, 3], index=[1, 2, 3], dtype=np.float),
                          pd.Series([1, 2, 3], index=[1, 2, 5], dtype=np.float))

    def test_series_compare_unsupported_types(self):
        self.assertRaises(YaUberSanityCheckError, sc.SanityChecker.series_compare,
                          [1, 2, 3],
                          [1, 2, 3])

    def test_series_compare_to_ser_str(self):
        s = sc.SanityChecker.to_ser_str(pd.Series([1, 2, 3], index=[1, 2, 3], dtype=np.float))
        self.assertTrue('pd.Series' in s)

    def test_report_test(self):
        s = sc.SanityChecker('test')
        self.assertRaises(YaUberSanityCheckError, s.report_test, 'test', 'test', True)
        s.tests['test'] = None

        s.report_test('test', 'test', True)
        s.report_test('test', 'test2', True)

        self.assertRaises(YaUberSanityCheckError, s.report_test, 'test', 'test', True)

        self.assertEqual(s.tests, {'test': {'test': True, 'test2': True}})

    def test_check_regular(self):
        with patch('yauber_algo.sanitychecks.SanityChecker.series_compare') as mock_series_compare:
            with patch('yauber_algo.sanitychecks.SanityChecker.report_test') as mock_report_test:
                mock_func = MagicMock()
                s = sc.SanityChecker('algo_test', debug=True).__enter__()

                expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.float)
                result = np.array([10, 20, 30, 40, 50, 60], dtype=np.float)
                mock_func.return_value = result

                # Default args
                s.check_regular(expected, mock_func, suffix='test')
                self.assertEqual(mock_func.called, True)
                self.assertEqual(mock_func.call_args[0], ())
                self.assertEqual(mock_func.call_args[1], {})

                # Custom args
                s.check_regular(expected, mock_func, (1,), {'test': True}, suffix='test')
                self.assertEqual(mock_func.called, True)
                self.assertEqual(mock_func.call_args[0], (1,))
                self.assertEqual(mock_func.call_args[1], {'test': True})

                self.assertEqual(mock_series_compare.called, True)
                self.assertEqual(mock_series_compare.call_args[0], (result, expected))

                self.assertEqual(mock_report_test.call_args[0], ('regular', 'test', True))

    def test_check_regular_expected_exception(self):
        with patch('yauber_algo.sanitychecks.SanityChecker.series_compare') as mock_series_compare:
            with patch('yauber_algo.sanitychecks.SanityChecker.report_test') as mock_report_test:
                mock_func = MagicMock()
                s = sc.SanityChecker('algo_test', debug=True).__enter__()

                expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.float)
                result = np.array([10, 20, 30, 40, 50, 60], dtype=np.float)

                def exception_valueerror_raised(*args, **kwargs):
                    raise ValueError()

                mock_func.side_effect = exception_valueerror_raised

                # Unexpected exception raised
                self.assertRaises(ValueError, s.check_regular, expected, mock_func, suffix='test')
                self.assertEqual(mock_report_test.called, False)

                # Expected exception
                s.check_regular(expected, mock_func, suffix='test', exception=ValueError)
                self.assertEqual(mock_report_test.called, True)

                # Expected but not Raised by code -> Assertion error
                mock_func.side_effect = lambda: result

                self.assertRaises(AssertionError, s.check_regular, expected, mock_func, suffix='test', exception=ValueError)



    def test_check_naninf(self):
        with patch('yauber_algo.sanitychecks.SanityChecker.series_compare') as mock_series_compare:
            with patch('yauber_algo.sanitychecks.SanityChecker.report_test') as mock_report_test:
                mock_func = MagicMock()
                s = sc.SanityChecker('algo_test', debug=True).__enter__()

                expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.float)
                result = np.array([10, 20, 30, 40, 50, 60], dtype=np.float)
                mock_func.return_value = result

                # Default args

                self.assertRaises(AssertionError, s.check_naninf, expected, mock_func, suffix='test')
                self.assertEqual(mock_func.called, True)
                self.assertEqual(mock_func.call_args[0], ())
                self.assertEqual(mock_func.call_args[1], {})

                # Custom args
                input_series = np.array([np.nan, 20, 30, 40, 50, 60], dtype=np.float)
                self.assertRaises(AssertionError, s.check_naninf, expected, mock_func, (input_series,), {'test': True}, suffix='test')
                self.assertEqual(mock_func.called, True)
                self.assertEqual(mock_func.call_args[0], (input_series,))
                self.assertEqual(mock_func.call_args[1], {'test': True})

                # Expected has inf
                expected2 = np.array([np.inf, 2, 3, 4, 5, 6], dtype=np.float)
                self.assertRaises(AssertionError, s.check_naninf, expected2, mock_func, (input_series,), {'test': True}, suffix='test')

                # Expected non float
                expected2 = np.array([1, 2, 3, 4, 5, 6], dtype=np.int)
                self.assertRaises(AssertionError, s.check_naninf, expected2, mock_func, (input_series,), {'test': True}, suffix='test')

                # Result must not have inf
                result2 = np.array([10, 20, np.nan, 40, np.inf, 60], dtype=np.float)
                mock_func.return_value = result2
                input_series = np.array([np.nan, 20, 30, np.inf, 50, 60], dtype=np.float)
                self.assertRaises(AssertionError, s.check_naninf, expected, mock_func, (input_series,), {'test': True}, suffix='test')

                # Series are not supported
                result2 = np.array([10, 20, np.nan, 40, np.inf, 60], dtype=np.float)
                mock_func.return_value = result2
                input_series = pd.Series([np.nan, 20, 30, np.inf, 50, 60], dtype=np.float)
                self.assertRaises(YaUberSanityCheckError, s.check_naninf, expected, mock_func, (input_series,), {'test': True}, suffix='test')


                # ALGOCODEX 2018-02-09 checks
                expected = np.array([np.nan, 2, 3, np.nan, 5, 6], dtype=np.float)
                result = np.array([np.nan, 20, 30, 50, 50, 60], dtype=np.float)
                mock_func.return_value = result
                input_series = np.array([np.nan, 20, 30, np.inf, 50, 60], dtype=np.float)
                self.assertRaises(AssertionError, s.check_naninf, expected, mock_func, (input_series,), {'test': True}, suffix='test')


                # Valid run
                expected = np.array([np.nan, 2, 3, np.nan, 5, 6], dtype=np.float)
                result = np.array([np.nan, 20, 30, np.nan, 50, 60], dtype=np.float)
                mock_func.return_value = result

                input_series = np.array([np.nan, 20, 30, np.inf, 50, 60], dtype=np.float)
                s.check_naninf(expected, mock_func, (input_series,), {'test': True}, suffix = 'test')
                self.assertEqual(mock_series_compare.called, True)
                self.assertEqual(mock_series_compare.call_args[0], (result, expected))
                self.assertEqual(mock_report_test.call_args[0], ('naninf', 'test', True))


    def test_check_series(self):
        with patch('yauber_algo.sanitychecks.SanityChecker.series_compare') as mock_series_compare:
            with patch('yauber_algo.sanitychecks.SanityChecker.report_test') as mock_report_test:
                mock_func = MagicMock()
                s = sc.SanityChecker('algo_test', debug=True).__enter__()

                expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.float)
                result = np.array([10, 20, 30, 40, 50, 60], dtype=np.float)
                mock_func.return_value = pd.Series(result)

                # Default args
                s.check_series(expected, mock_func, suffix='test')
                self.assertEqual(mock_func.called, True)
                self.assertEqual(mock_func.call_args[0], ())
                self.assertEqual(mock_func.call_args[1], {})

                # Custom args
                s.check_series(expected, mock_func, (1,), {'test': True}, suffix='test')
                self.assertEqual(mock_func.called, True)
                self.assertEqual(mock_func.call_args[0], (1,))
                self.assertEqual(mock_func.call_args[1], {'test': True})

                self.assertEqual(mock_series_compare.called, True)
                self.assertEqual(len(mock_series_compare.call_args[0]), 2)
                self.assertEqual(True, np.all(mock_series_compare.call_args[0][0].values == result))
                self.assertEqual(True, np.all(mock_series_compare.call_args[0][1] == expected))

                self.assertEqual(mock_report_test.call_args[0], ('series', 'test', True))

                # Assert must be series
                mock_func.return_value = result
                self.assertRaises(AssertionError, s.check_series, expected, mock_func, (1,), {'test': True}, suffix='test')

    def test_check_dtype_generic(self):
        with patch('yauber_algo.sanitychecks.SanityChecker.series_compare') as mock_series_compare:
            with patch('yauber_algo.sanitychecks.SanityChecker.report_test') as mock_report_test:
                mock_func = MagicMock()
                s = sc.SanityChecker('algo_test', debug=True).__enter__()

                expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.float)
                result = np.array([10, 20, 30, 40, 50, 60], dtype=np.float)
                mock_func.return_value = result

                # Default args
                self.assertRaises(AssertionError, s.check_dtype_generic, np.float, 'float', expected, mock_func, suffix='test')
                self.assertEqual(mock_func.called, True)
                self.assertEqual(mock_func.call_args[0], ())
                self.assertEqual(mock_func.call_args[1], {})

                # Custom args
                input_series = np.array([np.nan, 20, 30, np.inf, 50, 60], dtype=np.float)
                s.check_dtype_generic(np.float, 'dtype_float', expected, mock_func, (input_series,), {'test': True}, suffix='test')
                self.assertEqual(mock_func.called, True)
                self.assertEqual(mock_func.call_args[0], (input_series,))
                self.assertEqual(mock_func.call_args[1], {'test': True})

                self.assertEqual(mock_series_compare.called, True)
                self.assertEqual(mock_series_compare.call_args[0], (result, expected))

                self.assertEqual(mock_report_test.call_args[0], ('dtype_float', 'test', True))

    def test_check_dtype_float(self):
        with patch('yauber_algo.sanitychecks.SanityChecker.check_dtype_generic') as mock_check_dtype_generic:
            s = sc.SanityChecker('algo_test').__enter__()
            mock_func = MagicMock()
            expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.float)

            s.check_dtype_float(expected, mock_func, (1,), {'test': True}, suffix='test')

            self.assertEqual((np.float, 'dtype_float', expected, mock_func, (1,), {'test': True}), mock_check_dtype_generic.call_args[0])
            self.assertEqual({'suffix': 'test'}, mock_check_dtype_generic.call_args[1])

    def test_check_dtype_bool(self):
        with patch('yauber_algo.sanitychecks.SanityChecker.check_dtype_generic') as mock_check_dtype_generic:
            s = sc.SanityChecker('algo_test').__enter__()
            mock_func = MagicMock()
            expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.float)

            s.check_dtype_bool(expected, mock_func, (1,), {'test': True}, suffix='test')

            self.assertEqual((np.bool, 'dtype_bool', expected, mock_func, (1,), {'test': True}), mock_check_dtype_generic.call_args[0])
            self.assertEqual({'suffix': 'test'}, mock_check_dtype_generic.call_args[1])

    def test_check_dtype_int(self):
        with patch('yauber_algo.sanitychecks.SanityChecker.check_dtype_generic') as mock_check_dtype_generic:
            s = sc.SanityChecker('algo_test').__enter__()
            mock_func = MagicMock()
            expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.float)

            s.check_dtype_int(expected, mock_func, (1,), {'test': True}, suffix='test')

            self.assertEqual((np.int32, 'dtype_int', expected, mock_func, (1,), {'test': True}), mock_check_dtype_generic.call_args[0])
            self.assertEqual({'suffix': 'test'}, mock_check_dtype_generic.call_args[1])

    def test_check_dtype_object(self):
        with patch('yauber_algo.sanitychecks.SanityChecker.check_dtype_generic') as mock_check_dtype_generic:
            s = sc.SanityChecker('algo_test').__enter__()
            mock_func = MagicMock()
            expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.float)

            self.assertRaises(AssertionError, s.check_dtype_object, mock_func, (1,), {'test': True}, suffix='test')

            def raise_no_object(*args, **kwargs):
                raise YaUberAlgoDtypeNotSupportedError('test')

            mock_check_dtype_generic.side_effect = raise_no_object
            s.check_dtype_object(raise_no_object, (1,), {'test': True}, suffix = 'test')

            self.assertEqual({'suffix': 'test'}, mock_check_dtype_generic.call_args[1])

    def test_check_dtype_generic_exception(self):
            s = sc.SanityChecker('algo_test').__enter__()
            mock_func = MagicMock()

            def raise_no_object(*args, **kwargs):
                raise YaUberAlgoDtypeNotSupportedError('test')

            s.check_dtype_generic(np.float, 'dtype_float', (1,), raise_no_object, (1,), {'test': True},
                                  suffix='test',
                                  exception=YaUberAlgoDtypeNotSupportedError)

            # Rerise another exception
            self.assertRaises(YaUberAlgoDtypeNotSupportedError, s.check_dtype_generic,
                              np.float, 'dtype_float', (1,), raise_no_object, (1,), {'test': True},
                              suffix='test',
                              exception=ValueError)

    def test_check_fut_ref(self):
        with patch('yauber_algo.sanitychecks.SanityChecker.report_test') as mock_report_test:
            s = sc.SanityChecker('algo_test', debug=True).__enter__()

            def kwarg_ref(**kwargs):
                return ref(kwargs['arr'], kwargs['period'])

            # No fut ref
            s.check_futref(5, 1, ref, (sc.SAMPLE_10_FLOAT, -1), suffix='test')
            s.check_futref(5, 1, kwarg_ref, (), {'arr': sc.SAMPLE_10_FLOAT, 'period': -1})


            # Must raise fut-ref
            self.assertRaises(AssertionError, s.check_futref, 5, 1, ref, (sc.SAMPLE_10_FLOAT, 1))

            # Default args raise YaUberSanityCheckError (no dtype)
            self.assertRaises(YaUberSanityCheckError, s.check_futref, 5, 1, ref)

            # pd.Series not supported
            self.assertRaises(YaUberSanityCheckError, s.check_futref, 5, 1, ref, (pd.Series(sc.SAMPLE_10_FLOAT), -1))

            # Negative window, step params
            self.assertRaises(YaUberSanityCheckError, s.check_futref, -5, 1, ref, (sc.SAMPLE_10_FLOAT, -1))
            self.assertRaises(YaUberSanityCheckError, s.check_futref, 5, -1, ref, (sc.SAMPLE_10_FLOAT, -1))

            # Arg array < min_datapoints
            self.assertRaises(YaUberSanityCheckError, s.check_futref, 5, 1, ref, (sc.SAMPLE_10_FLOAT[:3], -1))
            self.assertRaises(YaUberSanityCheckError, s.check_futref, 5, 1, kwarg_ref, (), {'arr': sc.SAMPLE_10_FLOAT[:3], 'period': -1})

            # Min check count
            self.assertRaises(AssertionError, s.check_futref, 5, 1, ref, (sc.SAMPLE_10_FLOAT, -1), min_checks_count=100)

            # Check is reported
            s.check_futref(5, 1, ref, (sc.SAMPLE_10_FLOAT, -1), suffix='test')
            self.assertEqual(mock_report_test.call_args[0], ('futref', 'test', True))

            #
            # Check for expected future ref errors
            #
            setattr(sys.modules[ref.__module__], 'IS_WARN_FUTREF', True)
            setattr(sys.modules[ref.__module__], 'IS_RAISE_FUTREF', True)
            s.check_futref(5, 1, ref, (sc.SAMPLE_10_FLOAT, 1), expected=True)

            # Expected but not raised (warning)
            setattr(sys.modules[ref.__module__], 'IS_WARN_FUTREF', False)
            setattr(sys.modules[ref.__module__], 'IS_RAISE_FUTREF', True)
            self.assertRaises(AssertionError, s.check_futref, 5, 1, ref, (sc.SAMPLE_10_FLOAT, 1), expected=True)


            # Check for un-expected future ref errors
            setattr(sys.modules[ref.__module__], 'IS_WARN_FUTREF', True)
            setattr(sys.modules[ref.__module__], 'IS_RAISE_FUTREF', True)
            self.assertRaises(YaUberAlgoFutureReferenceError, s.check_futref, 5, 1, ref, (sc.SAMPLE_10_FLOAT, 1))


            # No warning assertion
            setattr(sys.modules[ref.__module__], 'IS_WARN_FUTREF', False)
            setattr(sys.modules[ref.__module__], 'IS_RAISE_FUTREF', True)
            self.assertRaises(AssertionError, s.check_futref, 5, 1, ref, (sc.SAMPLE_10_FLOAT, 1), expected=True)

            # No FutRef expected exception
            setattr(sys.modules[ref.__module__], 'IS_WARN_FUTREF', False)
            setattr(sys.modules[ref.__module__], 'IS_RAISE_FUTREF', False)
            self.assertRaises(YaUberSanityCheckError, s.check_futref, 5, 1, ref, (sc.SAMPLE_10_FLOAT, 1), expected=True)

            # Delayed fut ref
            def delayed_ref(arr, period):
                if len(arr) == len(sc.SAMPLE_10_FLOAT):
                    return ref(arr, period)

                if period > 0:
                    raise YaUberAlgoFutureReferenceError('Delayed fut ref TEST')
                else:
                    return ref(arr, period)

            self.assertRaises(YaUberSanityCheckError, s.check_futref, 5, 1, delayed_ref, (sc.SAMPLE_10_FLOAT, 1), expected = True)


    def test_check_window_consistency(self):
        with patch('yauber_algo.sanitychecks.SanityChecker.report_test') as mock_report_test:
            s = sc.SanityChecker('algo_test', debug=True).__enter__()

            def kwarg_ref(**kwargs):
                return ref(kwargs['arr'], kwargs['period'])

            def ref_consistency(arr, period):
                if period > 0:
                    if IS_WARN_FUTREF:
                        warnings.warn('Window incosistency modelled')

                    if IS_RAISE_FUTREF:
                        raise YaUberAlgoWindowConsistencyError()

                    return np.cumsum(arr)
                else:
                    return ref(arr, period)


            # No fut ref
            s.check_window_consistency(5, 1, ref, (sc.SAMPLE_10_FLOAT, -1), suffix='test')
            s.check_window_consistency(5, 1, kwarg_ref, (), {'arr': sc.SAMPLE_10_FLOAT, 'period': -1})


            # Must raise fut-ref
            self.assertRaises(AssertionError, s.check_window_consistency, 5, 1, np.cumsum, (sc.SAMPLE_10_FLOAT,))

            # Default args raise YaUberSanityCheckError (no dtype)
            self.assertRaises(YaUberSanityCheckError, s.check_window_consistency, 5, 1, ref)

            # pd.Series not supported
            self.assertRaises(YaUberSanityCheckError, s.check_window_consistency, 5, 1, ref, (pd.Series(sc.SAMPLE_10_FLOAT), -1))

            # Negative window, step params
            self.assertRaises(YaUberSanityCheckError, s.check_window_consistency, -5, 1, ref, (sc.SAMPLE_10_FLOAT, -1))
            self.assertRaises(YaUberSanityCheckError, s.check_window_consistency, 5, -1, ref, (sc.SAMPLE_10_FLOAT, -1))

            # Arg array < min_datapoints
            self.assertRaises(YaUberSanityCheckError, s.check_window_consistency, 5, 1, ref, (sc.SAMPLE_10_FLOAT[:3], -1))
            self.assertRaises(YaUberSanityCheckError, s.check_window_consistency, 5, 1, kwarg_ref, (), {'arr': sc.SAMPLE_10_FLOAT[:3], 'period': -1})

            # Min check count
            self.assertRaises(AssertionError, s.check_window_consistency, 5, 1, ref, (sc.SAMPLE_10_FLOAT, -1), min_checks_count=100)
            #s.check_window_consistency( 5, 1, ref, (sc.SAMPLE_10_FLOAT, -1), min_checks_count = 100)

            # Check is reported
            s.check_window_consistency(5, 1, ref, (sc.SAMPLE_10_FLOAT, -1), suffix='test', debug=True)
            self.assertEqual(mock_report_test.call_args[0], ('window_consistency', 'test', True))

            #
            # Check for expected future ref errors
            #
            IS_WARN_FUTREF = True
            IS_RAISE_FUTREF = True
            s.check_window_consistency(5, 1, ref_consistency, (sc.SAMPLE_10_FLOAT, 1), expected=True)

            # Expected but not raised (warning)
            IS_WARN_FUTREF = False
            IS_RAISE_FUTREF = True
            self.assertRaises(AssertionError, s.check_window_consistency, 5, 1, ref_consistency, (sc.SAMPLE_10_FLOAT, 1), expected=True)


            # Check for un-expected future ref errors
            IS_WARN_FUTREF = True
            IS_RAISE_FUTREF = True
            self.assertRaises(YaUberAlgoWindowConsistencyError, s.check_window_consistency, 5, 1, ref_consistency, (sc.SAMPLE_10_FLOAT, 1))


            # No warning assertion
            IS_WARN_FUTREF = False
            IS_RAISE_FUTREF = True
            self.assertRaises(AssertionError, s.check_window_consistency, 5, 1, ref_consistency, (sc.SAMPLE_10_FLOAT, 1), expected=True)

            # No FutRef expected exception
            IS_WARN_FUTREF = False
            IS_RAISE_FUTREF = False
            self.assertRaises(YaUberSanityCheckError, s.check_window_consistency, 5, 1, ref_consistency, (sc.SAMPLE_10_FLOAT, 1), expected=True)

            # Delayed fut ref
            def delayed_ref(arr, period):
                if len(arr) == len(sc.SAMPLE_10_FLOAT):
                    return ref_consistency(arr, period)

                if period > 0:
                    raise YaUberAlgoWindowConsistencyError('Delayed windows consistency TEST')
                else:
                    return ref_consistency(arr, period)

            self.assertRaises(YaUberSanityCheckError, s.check_window_consistency, 5, 1, delayed_ref, (sc.SAMPLE_10_FLOAT, 1),
                              expected = True, debug=True)


