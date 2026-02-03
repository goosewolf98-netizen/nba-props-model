import unittest
import math
from src.probability import calculate_probs

class TestProbability(unittest.TestCase):
    def test_poisson_calculation(self):
        # Stat: reb, Line: 10.5, Proj: 10.0
        # P(Under) = CDF(10.5, 10.0) -> CDF(10, 10)
        res = calculate_probs('reb', 10.0, 10.5)
        self.assertAlmostEqual(res['p_over'] + res['p_under'], 1.0)
        self.assertGreater(res['p_under'], 0.5) # Since proj < line

    def test_normal_calculation(self):
        # Stat: pts, Line: 20.5, Proj: 20.5, RMSE: 5.0
        res = calculate_probs('pts', 20.5, 20.5, rmse=5.0)
        self.assertAlmostEqual(res['p_over'], 0.5)
        self.assertAlmostEqual(res['p_under'], 0.5)
        self.assertEqual(res['rmse_used'], 5.0)

    def test_normal_fallback(self):
        # Stat: pts, Line: 20.5, Proj: 25.0, RMSE: None -> Default 1.0
        res = calculate_probs('pts', 25.0, 20.5)
        self.assertEqual(res['rmse_used'], 1.0)
        # Proj (25) > Line (20.5) -> Over should be high
        self.assertGreater(res['p_over'], 0.99)

if __name__ == '__main__':
    unittest.main()
