import unittest
import pandas as pd
import numpy as np
from ..pymcDoseResponse import doseResponseModel, loadCellTiter, loadIncucyte


class TestDoseResponseMethods(unittest.TestCase):
    def test_loadCellTiter(self):

        data = loadCellTiter(drug='DOX')

        self.assertIsInstance(data, pd.DataFrame)

    def test_loadIncucyte(self):

        data = loadIncucyte()

        self.assertIsInstance(data, pd.DataFrame)

    def test_doseResponseModel(self):

        M = doseResponseModel()

        self.assertIsInstance(M.drugCs, np.ndarray)


