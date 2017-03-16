import random
import time
import unittest
import numpy as np
from ..GrowthModel import rate_values

class TestgrMethods(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f" % (self.id(), t*1000))

    def test_rate_pnumbers(self):
        input = [[1, 2], [3, 4], [5, 6]]

        output = rate_values(input, 1.0)

        # Test that the output list of parameters equals the expected length
        self.assertEqual(len(output), 3)

        # Test that the parameters output are all positive
        for ii in range(len(output)):
            self.assertGreaterEqual(output[ii], 0)

        # Test that the function raises an exception with a negative time
        # TODO: Insert here.

        # What should happen with, say, an empty list?
        # TODO: Insert here.

if __name__ == '__main__':
    unittest.main()
