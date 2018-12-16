import unittest
from unittest.mock import MagicMock, patch
from yauber_algo.algo import twma_weights_exponential, ema
import pandas as pd
import numpy as np


class EMATestCase(unittest.TestCase):
    def test_twma_weights_exponential(self):

        w = twma_weights_exponential(10)

        _alpha = 2 / (10 + 1)
        expected_w = (1 - _alpha) ** np.arange(10)
        self.assertEqual(True, np.all(expected_w == w))

        # Custom alpha
        w = twma_weights_exponential(10, alpha=0.5)
        _alpha = 0.5
        expected_w = (1 - _alpha) ** np.arange(10)
        self.assertEqual(True, np.all(expected_w == w))

    def test_ema(self):
        with patch('yauber_algo.algo.twma_weights_exponential') as mock_twma_w_exp:
            with patch('yauber_algo.algo.twma') as mock_twma:
                a = np.array([1,2, 3, 4, 5])
                mock_twma_w_exp.return_value = [1, 2, 3]
                mock_twma.return_value = [4, 5, 6]

                result = ema(a, 10)
                self.assertEqual((10,), mock_twma_w_exp.call_args[0])
                self.assertEqual({'alpha': None}, mock_twma_w_exp.call_args[1])

                self.assertEqual((a, [1, 2, 3]), mock_twma.call_args[0])
                self.assertEqual([4, 5, 6], result)





if __name__ == '__main__':
    unittest.main()
