import unittest
import numpy as np


class TestgrMethods(unittest.TestCase):
    def setUp(self):
        from ..pymcGrowth import GrowthModel

        self.GR = GrowthModel()

#    def test_oldmodel(self):
#        """ Test logL run """
#        params = np.power(10,[-2,-3,-3,-3,])
#        conv = np.power(10,0.9)
#        self.GR.old_model(params, conv)
    
    def test_bothmodels(self):
        self.GR.importData(3)
        df = self.GR.sample()
        df = df.sample(10)
        for row in df.iterrows():
            params = row[1].as_matrix()[0:4]
            conv = row[1].as_matrix()[4]
            self.assertAlmostEqual(row[1].as_matrix()[5],self.GR.old_model(params,conv),delta = row[1].as_matrix()[5] / 10**6)


#    def test_integral_data(self):
#        """ TODO: describe test """
#        from ..GrowthModel import simulate
#
#        params = np.array([0.009, 0.016, 0.01, 0.008])
#        t_interval = np.arange(0, 10, .005)
#
#        output = simulate(params, t_interval)
#
#        # Test that we get back time and output
#        self.assertEqual(len(output), 2000)


if __name__ == '__main__':
    unittest.main()