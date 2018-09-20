''' Test the least squares drug interaction fitting. '''
import unittest
from ..lsqFitting import lsqFitting


class TestLsqFitting(unittest.TestCase):
    @unittest.skip('Least squares fitting should be depracated as soon as it works through MCMC.')
    def test_lsq_fitting(self):
        """ Test that we can successfully find the fitted parameters with small cost """
        ssr = lsqFitting(hold=True)[1]

        self.assertLess(ssr, 10., "Find fitted parameters with a hold as 0.0")

        ssr = lsqFitting(hold=False)[1]

        self.assertLess(ssr, 0.5, "Does not find fitted parameters with varifying a")
