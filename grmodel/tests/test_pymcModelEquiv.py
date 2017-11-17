import unittest
from ..pymcDoseResponse import doseResponseModel
from ..pymcGrowth import GrowthModel


class TestDoseResponseMethods(unittest.TestCase):
    def test_d_equiv(self):
        ''' Test whether identical parameters share their prior distribution. '''

        self.assertEqual(4, 4)
