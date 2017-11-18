import unittest
from ..figures.Figure1 import doseResponseTiter


class TestDoseResponseMethods(unittest.TestCase):
    def test_CellTiterFigure(self):

        doseResponseTiter()

        self.assertEqual(4, 4)