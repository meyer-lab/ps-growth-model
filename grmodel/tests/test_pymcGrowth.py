import unittest
import numpy as np


class TestgrMethods(unittest.TestCase):
    def setUp(self):
        from ..pymcGrowth import GrowthModel

        self.GR = GrowthModel()

    def test_oldmodel(self):
        """ Test logL run """
        self.GR.importData(3)
        params = np.power(10,[-1.5,-3,-3,-3])
        conv = np.power(10,0)
        apopcon = np.power(10,-0.8)
        dnacon = np.power(10,-0.8)
        print(self.GR.old_model(params, conv, apopcon, dnacon)[0])

    def test_bothmodels(self):
        from ..utils import read_dataset 
        classM, df = read_dataset(3)
        df = df.sample(10)
        for row in df.iterrows():
            params = row[1].as_matrix()[0:4]
            conv = row[1]['confl_conv']
            apopcon = row[1]['apop_conv']
            dnacon = row[1]['dna_conv']
            self.assertAlmostEqual(row[1].as_matrix()[-4], 
                                   classM.old_model(params,conv,apopcon,dnacon)[0],
                                   delta = row[1].as_matrix()[-4] / 10**6)

    def test_model(self):
        self.GR.importData(3)
        self.GR.getMAP()

    def test_sim_plot(self):
        from ..utils import sim_plot
        sim_plot(3)
        sim_plot(13)
        sim_plot(2)

    def test_hist_plot(self):
        from ..utils import hist_plot
        hist_plot([2,3,4,5,6])

    def test_fit_plot(self):
        from ..utils import fit_plot
        params = np.exp([-3.4820022279157254, -4.618086088299654, -4.673132496944741, -4.993737331758884, -0.7971962401826481, -2.5886341322921487, -2.5888985087520373])
        fit_plot(params, 3)

    def test_dose_response_plots(self):
        from ..utils import dose_response_plot, violinplot
        dose_response_plot(['Dox','NVB'], log = True)
        violinplot(['Dox','NVB'], log = True)

    def test_062117(self):
        from ..utils import dose_response_plot, violinplot
        dose_response_plot(['U0126','JNK-IN-7','R428','Erlotinib'], log = True)
        violinplot(['U0126','JNK-IN-7', 'R428', 'Erlotinib'], log = True)

    def test_PCA(self):
        from ..utils import PCA
        PCA([2,3,4,5,6])

if __name__ == '__main__':
    unittest.main()