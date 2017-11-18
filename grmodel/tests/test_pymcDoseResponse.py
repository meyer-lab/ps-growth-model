import unittest
import pandas as pd
from ..pymcDoseResponse import doseResponseModel, loadCellTiter


class TestDoseResponseMethods(unittest.TestCase):
    def test_loadCellTiter(self):

        data = loadCellTiter()

        self.assertIsInstance(data, pd.DataFrame)

