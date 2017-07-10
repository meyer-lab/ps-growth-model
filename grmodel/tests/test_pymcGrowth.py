import unittest
import numpy as np


class TestgrMethods(unittest.TestCase):
    def setUp(self):
        from ..pymcGrowth import GrowthModel

        self.GR = GrowthModel()

    def test_oldmodel(self):
        """ Test logL run """
        self.GR.importData(3)
        params = [0.023756, 2739.119062, 3.041182e+00, 1.318788e-04]
        conv = 1.116687
        self.GR.old_model(params, conv)[0]
        parms = np.power(10,[-1.5,-4,-4,-3])
        con = np.power(10,0.75)
        self.GR.old_model(parms, con)[0]

    def test_bothmodels(self):
        from ..utils import read_dataset 
        classM, df = read_dataset(3)
        df = df.sample(10)
        for row in df.iterrows():
            params = row[1].as_matrix()[0:4]
            conv = row[1]['confl_conv']
            self.assertAlmostEqual(row[1].as_matrix()[5], 
                                   classM.old_model(params,conv)[0],
                                   delta = row[1].as_matrix()[5] / 10**6)

    def test_model(self):
        self.GR.importData(3)
        self.GR.getMAP()

    def test_sim_plot(self):
        from ..utils import sim_plot
        sim_plot(2)

    def test_hist_plot(self):
        from ..utils import hist_plot
        hist_plot()

    def test_map_plot(self):
        from ..utils import fit_plot
        fit_plot([-3.434892865275417,-7.96312272337698,-7.987219333825418,-5.2248634220018575,1.7561011061148422], 3)

    def test_dose_response_plots(self):
        from ..utils import dose_response_plot, violinplot
        dose_response_plot(['Dox','NVB'], log = True)
        violinplot(['Dox','NVB'])

if __name__ == '__main__':
    unittest.main()