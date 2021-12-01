from unittest import TestCase
import numpy as np

from utils import ChamferDistance, chamfer_distance


class TestChamferDistance(TestCase):
    def test__array2samples_distance(self):
        a = np.random.random((20, 3))
        b = np.random.random((23, 3))
        cd = ChamferDistance()(a, b)
        print(cd)
        cd = chamfer_distance(a.reshape((1, 20, 3)), b.reshape((1, 23, 3)))
        print(cd)
    def test__pointcloudsize_distance(self):
        a = np.random.random((20, 3))
        b = np.random.random((20, 3))
        cd = ChamferDistance()(a, b)
        print(cd)
        a = np.random.random((40, 3))
        b = np.random.random((40, 3))
        cd = ChamferDistance()(a, b)
        print(cd)
