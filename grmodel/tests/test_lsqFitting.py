import unittest
from ..lsqFitting import lsqFitting


class TestLsqFitting(unittest.TestCase):
    def test_lsq_fitting(self):
        """ Test that we can successfully find the fitted parameters with small ssr(sum of squares residuals) """
        ssr = lsqFitting(hold=True)[1]

        self.assertGreater(ssr, 2.0, "Find fitted parameters with a hold as 0.0")

        ssr = lsqFitting(hold=False)[1]

        self.assertLess(ssr, 0.5, "Does not find fitted parameters with varifying a")
