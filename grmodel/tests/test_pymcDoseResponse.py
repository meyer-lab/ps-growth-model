import unittest
import pandas as pd
import numpy as np
from ..pymcDoseResponse import doseResponseModel, loadCellTiter


class TestDoseResponseMethods(unittest.TestCase):
    """ Tests the model of just using live cell number. """

    def test_loadCellTiter(self):
        """ Test that we successfully load the CellTiter Glo data. """

        data = loadCellTiter(drug="DOX")

        self.assertIsInstance(data, pd.DataFrame)

    def test_doseResponseModel(self):
        """ Test that we can successfully build the model. """
        M = doseResponseModel()

        self.assertIsInstance(M.drugCs, np.ndarray)
