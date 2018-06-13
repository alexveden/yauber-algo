import unittest
from unittest.mock import patch
from yauber_algo.errors import *
from yauber_algo.algo import median
import numpy as np

class MedianTestCase(unittest.TestCase):
    def test_median(self):
        a = [1, 2, 3]

        with patch('yauber_algo.algo.quantile') as mock_quantile:
            mock_quantile.return_value = [666, 777, 888]

            res = median(a, 6)

            self.assertEqual(mock_quantile.called, True)
            self.assertEqual(mock_quantile.call_args[0], (a, 6, 0.5))
            self.assertEqual(res, [666, 777, 888])

