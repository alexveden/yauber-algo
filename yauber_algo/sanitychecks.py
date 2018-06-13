"""
Sanity check module helps automatically test the algos for future ref issues, NaN handling, back history shrinking
"""
import pandas as pd
import numpy as np
from yauber_algo.errors import YaUberSanityCheckError, YaUberAlgoFutureReferenceError, YaUberAlgoWindowConsistencyError, YaUberAlgoDtypeNotSupportedError
from io import StringIO
from unittest.mock import patch

TESTS_NAMES = ['regular', 'series', 'naninf', 'dtype_float', 'dtype_bool', 'dtype_int', 'dtype_object', 'futref', 'window_consistency']

#
# Sample data
#
SAMPLE_10_FLOAT = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float)
SAMPLE_10S_FLOAT = pd.Series(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), index=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]), dtype=np.float)
SAMPLE_10_NANINF = np.array([1, 2, np.nan, 4, 5, 6, np.inf, 8, 9, 10], dtype=np.float)
SAMPLE_10_BOOL = np.array([True, True, False, False, True, True, False, False, True, True], dtype=np.bool)
SAMPLE_10_INT = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
SAMPLE_10_OBJ = np.array([True, True, False, False, True, np.nan, False, False, True, True], dtype=np.object)


class SanityChecker:
    def __init__(self, algo_name, **kwargs):
        self.debug = kwargs.get('debug', False)
        self.algo_name = algo_name
        self.tests = {}
        self.tests_suppressed = set()

    def __enter__(self):
        self.tests = {test_name: None for test_name in TESTS_NAMES}
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Check all tests if only no exception occurred after sanity checks
            buf = StringIO()

            has_failed = False
            buf.write('------------------------------------\n')
            buf.write(f'Sanity tests for: {self.algo_name}\n')
            buf.write('------------------------------------\n')
            for t_name, t_result in self.tests.items():
                if t_result is None:
                    if t_name not in self.tests_suppressed:
                        has_failed = True
                        buf.write('{0:<30}[MISS]\n'.format(t_name))
                    else:
                        buf.write('{0:<30}[SKIP]\n'.format(t_name))
                else:
                    for k, v in t_result.items():
                        _name = t_name+"_"+k if k else t_name
                        if v:
                            buf.write('{0:<30}[ OK ]\n'.format(_name))
                        else:
                            buf.write('{0:<30}[FAIL]\n'.format(_name))

            if has_failed:
                print(buf.getvalue())
                raise YaUberSanityCheckError("Sanity checker has missed or failed tests!")
            elif self.debug:
                print(buf.getvalue())

    def suppress_test(self, test_name):
        if test_name not in TESTS_NAMES:
            raise ValueError(f'Trying to suppress unknown test: {test_name}')

        self.tests_suppressed.add(test_name)

    @staticmethod
    def series_compare(algo_result, expected):
        if type(algo_result) != type(expected):
            raise AssertionError(f'Series must have the same type: {type(algo_result)} != {type(expected)}')
        if len(algo_result) != len(expected):
            raise AssertionError(f'Series must have the same length: {len(algo_result)} != {len(expected)}')

        if isinstance(algo_result, np.ndarray):
            if algo_result.dtype != np.float64:
                raise AssertionError(f'Algo result must always return np.float64 dtype')

            if algo_result.dtype != expected.dtype:
                raise AssertionError(f'Series must have the same d-type: {algo_result.dtype} != {expected.dtype}')

            if not np.allclose(algo_result, expected, equal_nan=True):
                raise AssertionError(f'Series values are not equal\n'
                                     f'Algo    : {algo_result}\n'
                                     f'Expected: {expected}')
            if sum(np.isinf(algo_result)) > 0:
                raise AssertionError(f'Algo results must not return Inf')

        elif isinstance(algo_result, pd.Series):
            if algo_result.dtype != np.float64:
                raise AssertionError(f'Algo result must always return np.float64 dtype')
            if algo_result.dtype != expected.dtype:
                raise AssertionError(f'Series must have the same d-type: {type(algo_result.dtype)} != {type(expected.dtype)}')
            if not np.all(algo_result.index == expected.index):
                raise AssertionError(f'Series have different indexes')
            if not np.allclose(algo_result.values, expected.values, equal_nan=True):
                raise AssertionError(f'Series values are not equal\n'
                                     f'Algo    : {algo_result.values}\n'
                                     f'Expected: {expected.values}')
            if sum(np.isinf(algo_result.values)) > 0:
                raise AssertionError(f'Algo results must not return Inf')
        else:
            raise YaUberSanityCheckError("Algo results must be pd.Series or np.ndarray")

    @staticmethod
    def to_ser_str(series):
        return f"""
        pd.Series({list(series.values)},
                  index={list(series.index.values)}),
        """

    def report_test(self, test_name, test_suffix, result):
        if test_name not in self.tests:
            raise YaUberSanityCheckError("The test name is not found in TESTS_NAMES or you are running SanityChecker out of 'with' context")

        if self.tests[test_name] is None:
            self.tests[test_name] = {test_suffix: result}
        else:
            if test_suffix in self.tests[test_name]:
                raise YaUberSanityCheckError(f"Duplicate suffix '{test_suffix}' for test type '{test_name}'")
            self.tests[test_name][test_suffix] = result

    def check_regular(self, expected, func, f_args=None, f_kwargs=None, **kwargs):
        if f_args is None:
            f_args = ()
        if f_kwargs is None:
            f_kwargs = {}

        expected_exception = kwargs.get('exception', None)

        try:
            result = func(*f_args, **f_kwargs)
        except Exception as exc:
            if expected_exception:
                if isinstance(exc, expected_exception):
                    self.report_test('regular', kwargs.get('suffix', ''), True)
                    return
            raise exc

        if expected_exception:
            raise AssertionError(f'Expected exception {expected_exception} was not raised!')

        self.series_compare(result, expected)
        self.report_test('regular', kwargs.get('suffix', ''), True)

    def check_naninf(self, expected, func, f_args=None, f_kwargs=None, **kwargs):
        if f_args is None:
            f_args = ()
        if f_kwargs is None:
            f_kwargs = {}

        result = func(*f_args, **f_kwargs)

        has_nan = False
        has_inf = False
        for a in list(f_args) + list(f_kwargs.values()):
            if isinstance(a, np.ndarray):
                if sum(np.isinf(a)) > 0:
                    has_inf = True
                if sum(np.isnan(a)) > 0:
                    has_nan = True
            elif isinstance(a, pd.Series):
                raise YaUberSanityCheckError("Only np.arrays are allowed in this test")

        if expected.dtype != np.float:
            raise AssertionError(f'All algos of yauber_algo.algo must return dtype=np.float, you expect: {expected.dtype}')
        if sum(np.isinf(expected)) > 0:
            raise AssertionError(f'Result of the algorithm must never return Inf values, EXPECTED Inf values must be replaced by NaNs!')
        if not has_nan:
            raise AssertionError(f'At least one of f_args must be array with NaN values')
        if not has_inf:
            raise AssertionError(f'At least one of f_args must be array with Inf values')

        self.series_compare(result, expected)
        if sum(np.isinf(result)) > 0:
            raise AssertionError(f'Result of the algorithm must never return Inf values, Inf values must be replaced by NaNs!')

        for a in list(f_args) + list(f_kwargs.values()):
            if isinstance(a, np.ndarray):
                if np.any(np.logical_and(~np.isnan(result), ~np.isfinite(a))):
                    if not kwargs.get('ignore_nan_argument_position_check', False):
                        raise AssertionError(f'Result NaN position of argument array must always have NaN in results (see ALGOCODEX 2018-02-09)\n'
                                         f'Result  : {result}\n'
                                         f'Argument: {a}')


        self.report_test('naninf', kwargs.get('suffix', ''), True)

    def check_series(self, expected, func, f_args=None, f_kwargs=None, **kwargs):
        if f_args is None:
            f_args = ()
        if f_kwargs is None:
            f_kwargs = {}

        result = func(*f_args, **f_kwargs)
        self.series_compare(result, expected)
        if not isinstance(result, pd.Series):
            raise AssertionError("Result must be pd.Series")

        self.report_test('series', kwargs.get('suffix', ''), True)

    def check_dtype_bool(self, expected, func, f_args=None, f_kwargs=None, **kwargs):
        self.check_dtype_generic(np.bool, 'dtype_bool', expected, func, f_args, f_kwargs, **kwargs)

    def check_dtype_int(self, expected, func, f_args=None, f_kwargs=None, **kwargs):
        self.check_dtype_generic(np.int32, 'dtype_int', expected, func, f_args, f_kwargs, **kwargs)

    def check_dtype_float(self, expected, func, f_args=None, f_kwargs=None, **kwargs):
        self.check_dtype_generic(np.float, 'dtype_float', expected, func, f_args, f_kwargs, **kwargs)

    def check_dtype_object(self, func, f_args=None, f_kwargs=None, **kwargs):
        try:
            self.check_dtype_generic(np.object, 'dtype_object', np.array([]), func, f_args, f_kwargs, **kwargs)
            raise AssertionError("Expected the YaUberAlgoDtypeNotSupportedError exception, because np.object is not supported DTYPE by design!")
        except YaUberAlgoDtypeNotSupportedError:
            self.report_test('dtype_object', kwargs.get('suffix', ''), True)


    def check_dtype_generic(self, dtype, check_name, expected, func, f_args=None, f_kwargs=None, **kwargs):
        if f_args is None:
            f_args = ()
        if f_kwargs is None:
            f_kwargs = {}

        expected_exception = kwargs.get('exception', None)

        try:
            result = func(*f_args, **f_kwargs)
        except Exception as exc:
            if expected_exception:
                if isinstance(exc, expected_exception):
                    self.report_test(check_name, kwargs.get('suffix', ''), True)
                    return
            raise exc

        has_dtype = False
        for a in list(f_args) + list(f_kwargs.values()):
            if isinstance(a, (np.ndarray, pd.Series)):
                if a.dtype == dtype:
                    has_dtype = True

        if not has_dtype:
            raise AssertionError(f'At least one of f_args must be array with dtype={dtype}')

        self.series_compare(result, expected)

        self.report_test(check_name, kwargs.get('suffix', ''), True)

    def check_futref(self, min_datapoints, step, func, f_args=None, f_kwargs=None, **kwargs):
        if f_args is None:
            f_args = ()
        if f_kwargs is None:
            f_kwargs = {}

        has_ndarray = False
        for a in list(f_args) + list(f_kwargs.values()):
            if isinstance(a, np.ndarray):
                has_ndarray = True
            elif isinstance(a, pd.Series):
                raise YaUberSanityCheckError("Only np.arrays are allowed in this test")

        if not has_ndarray:
            raise YaUberSanityCheckError(f'To perform fut_ref rest we need at least 1 np.array in f_args')

        is_expected_fut_ref = kwargs.get('expected', False)
        min_checks_count = kwargs.get('min_checks_count', 5)
        fut_ref_error_raised = False
        fut_ref_warn = False

        if step <= 0:
            raise YaUberSanityCheckError("Step must be > 0")
        if min_datapoints <= 0:
            raise YaUberSanityCheckError('min_datapoints must be > 0')

        # Calculate function on full history
        with patch('warnings.warn') as mock_warn:
            try:
                result_full = func(*f_args, **f_kwargs)
            except YaUberAlgoFutureReferenceError as exc:
                if is_expected_fut_ref:
                    if not mock_warn.called:
                        raise AssertionError('Expected to get Fut Ref warning message')
                    self.report_test('futref', kwargs.get('suffix', ''), True)
                    # Expected behavior

                    return
                else:
                    raise exc

        hist_len = min_datapoints
        checks_count = 0

        while True:
            f_args_alt = []
            f_kwargs_alt = {}

            is_break = False

            for a in f_args:
                if isinstance(a, np.ndarray):
                    if len(a) < min_datapoints:
                        raise YaUberSanityCheckError("Length of f_args array less than min_datapoints")

                    if hist_len >= len(a):
                        is_break = True
                    else:
                        f_args_alt.append(a[:hist_len])
                else:
                    f_args_alt.append(a)

            for k, a in f_kwargs.items():
                if isinstance(a, np.ndarray):
                    if len(a) < min_datapoints:
                        raise YaUberSanityCheckError("Length of f_kwargs array less than min_datapoints")

                    if hist_len >= len(a):
                        is_break = True
                    else:
                        f_kwargs_alt[k] = a[:hist_len]
                else:
                    f_kwargs_alt[k] = a

            if is_break:
                break

            # Compare results
            with patch('warnings.warn') as mock_warn:
                try:
                    result_sample = func(*f_args_alt, **f_kwargs_alt)
                    if not np.allclose(result_full[:hist_len], result_sample, equal_nan=True):
                        raise AssertionError('Future reference detected')
                except YaUberAlgoFutureReferenceError as exc:
                    raise YaUberSanityCheckError('Future ref detected but YaUberAlgoFutureReferenceError not raised at the first step')
                except AssertionError as exc:
                    if is_expected_fut_ref:
                        raise YaUberSanityCheckError("Algorithm must raise YaUberAlgoFutureReferenceError if 'is_expected_fut_ref'")
                    else:
                        raise exc

            hist_len += step
            checks_count += 1

        if checks_count < min_checks_count:
            raise AssertionError(f"Future ref must perform at least "
                                 f"'min_checks_count' > {min_checks_count} calculations")

        self.report_test('futref', kwargs.get('suffix', ''), True)

    def check_window_consistency(self, min_datapoints, step, func, f_args=None, f_kwargs=None, **kwargs):
        if f_args is None:
            f_args = ()
        if f_kwargs is None:
            f_kwargs = {}

        has_ndarray = False
        for a in list(f_args) + list(f_kwargs.values()):
            if isinstance(a, np.ndarray):
                has_ndarray = True
            elif isinstance(a, pd.Series):
                raise YaUberSanityCheckError("Only np.arrays are allowed in this test")

        if not has_ndarray:
            raise YaUberSanityCheckError(f'To perform window_consistency rest we need at least 1 np.array in f_args')

        is_expected_fut_ref = kwargs.get('expected', False)
        min_checks_count = kwargs.get('min_checks_count', 5)
        fut_ref_error_raised = False
        fut_ref_warn = False

        if step <= 0:
            raise YaUberSanityCheckError("Step must be > 0")
        if min_datapoints <= 0:
            raise YaUberSanityCheckError('min_datapoints must be > 0')

        # Calculate function on full history
        with patch('warnings.warn') as mock_warn:
            try:
                result_full = func(*f_args, **f_kwargs)
            except YaUberAlgoWindowConsistencyError as exc:
                if is_expected_fut_ref:
                    if not mock_warn.called:
                        raise AssertionError('Expected to get window consistency warning message')
                    self.report_test('window_consistency', kwargs.get('suffix', ''), True)
                    # Expected behavior
                    return
                else:
                    raise exc

        hist_len = min_datapoints
        checks_count = 0

        while True:
            f_args_alt = []
            f_kwargs_alt = {}

            is_break = False

            for a in f_args:
                if isinstance(a, np.ndarray):
                    if len(a) < min_datapoints:
                        raise YaUberSanityCheckError("Length of f_args array less than min_datapoints")

                    if hist_len >= len(a):
                        is_break = True
                    else:
                        f_args_alt.append(a[-hist_len:])
                else:
                    f_args_alt.append(a)

            for k, a in f_kwargs.items():
                if isinstance(a, np.ndarray):
                    if len(a) < min_datapoints:
                        raise YaUberSanityCheckError("Length of f_kwargs array less than min_datapoints")

                    if hist_len >= len(a):
                        is_break = True
                    else:
                        f_kwargs_alt[k] = a[-hist_len:]
                else:
                    f_kwargs_alt[k] = a

            if is_break:
                break

            # Compare results
            with patch('warnings.warn') as mock_warn:
                try:
                    result_sample = func(*f_args_alt, **f_kwargs_alt)
                    if kwargs.get('debug'):
                        print(f'Step: {step}')
                        print(f'Hist Len: {hist_len}')
                        print(f'smpl: {result_sample}')
                        print(f'full: {result_full[-hist_len:]}')

                    is_not_nan = ~np.isnan(result_sample)
                    full_slice = result_full[-hist_len:][is_not_nan]
                    sample_slice = result_sample[is_not_nan]
                    if not np.allclose(full_slice, sample_slice, equal_nan=True):
                        result_sample = func(*f_args_alt, **f_kwargs_alt)
                        raise AssertionError('Window inconsistency detected!\n' 
                                             'Make sure that func() period > min_datapoints\n' 
                                             f'Full    : {result_full[-hist_len:]}\n' 
                                             f'Windowed: {result_sample}')
                except YaUberAlgoWindowConsistencyError as exc:
                    raise YaUberSanityCheckError('Window inconsistency detected but YaUberAlgoWindowConsistencyError not raised at the first step')
                except AssertionError as exc:
                    if is_expected_fut_ref:
                        raise YaUberSanityCheckError("Algorithm must raise YaUberAlgoWindowConsistencyError if 'is_expected_fut_ref'")
                    else:
                        raise exc

            hist_len += step
            checks_count += 1

        if checks_count < min_checks_count:
            raise AssertionError(f"Window consistency must perform at least " 
                                 f"'min_checks_count' > {min_checks_count} calculations")

        self.report_test('window_consistency', kwargs.get('suffix', ''), True)










