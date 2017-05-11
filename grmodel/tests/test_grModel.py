import time
import unittest
import pandas
import numpy as np
from ..GrowthModel import rate_values, GrowthModel, mcFormat, simulate, ODEfun

class TestgrMethods(unittest.TestCase):
    def setUp(self):
        self.GR = GrowthModel(selCol=5)

    def test_load_data(self):
        """ Test data import. """

        #test to make sure returned object is pandas DataFrame
        self.assertTrue(isinstance(self.GR.data_confl, pandas.core.frame.DataFrame))

        #test to make sure returned object is pandas DataFrame
        self.assertTrue(isinstance(self.GR.data_green, pandas.core.frame.DataFrame))

    def test_logL(self):
        """ Test logL run """
        self.GR.logL(self.GR.lb)


    def test_rate_pnumbers(self):
        """ TODO: describe test """
        inputt = np.full((3, 5), 1.0)

        output = rate_values(inputt, 1.0)

        # Test that the function raises an exception with empty
        with self.assertRaises(AttributeError):
            rate_values([], 5)

        # Test that the output list of parameters equals the expected length
        self.assertEqual(len(output), 5)

        # Test that the parameters output are all positive
        for _, item in enumerate(output):
            self.assertGreaterEqual(item, 0)

        # Test that the function raises an exception with a negative time
        with self.assertRaises(ValueError):
            rate_values(inputt, -1.0)

    def test_mcFormat(self):
        """ TODO: describe test """
        inputt = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        output = mcFormat(inputt)

        #Test that output list of lists is correct length
        self.assertEqual(output.shape, (2, 5))

    def test_ODE(self):
        """ TODO: describe test """
        inputt = mcFormat(np.array([0.0009, -0.016, 0.01, 0.008, 0.0007, 0.005, -0.001, -0.0071, 0.0008, 0.005]))
        input_state = [100, 10, 10, 5]

        output = ODEfun(input_state, 2.0, inputt)
        correct = [-106.771976782, 102.37465891, 91.226595602, 10.108585305]

        #test that it outputs 4 rates
        self.assertEqual(len(output), 4)

        #test that outputs are in realistic range
        for i, item in enumerate(output):
            self.assertAlmostEqual(item, correct[i], 1) # last # is # of places we're rounding to

        #if we're given a negative time
        with self.assertRaises(ValueError):
            ODEfun(input_state, -2.0, inputt)

        #if params is an empty list
        with self.assertRaises(AttributeError):
            ODEfun(input_state, 2.0, [])

    def test_integral_data(self):
        """ TODO: describe test """
        params = mcFormat(np.array([0.0009, -0.016, 0.01, 0.008, 0.0007, 0.005, -0.001, -0.0071, 0.0008, 0.005]))
        t_interval = np.arange(0, 10, .005)

        output = simulate(params, t_interval)

        #test to make sure returned object is pandas DataFrame
        self.assertTrue(isinstance(output, pandas.core.frame.DataFrame))

        #test to make sure there are 5 columns in the dataframe
        self.assertEqual(len(output.keys()), 5)

        #test to make sure each column is correct length
        self.assertEqual(len(output), len(t_interval))


if __name__ == '__main__':
    unittest.main()
