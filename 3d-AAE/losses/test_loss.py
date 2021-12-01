import unittest

import torch
import numpy as np

from champfer_loss import ChamferLoss
from chamfer_loss import CustomChamferDistance


class MyTestCase(unittest.TestCase):
    def test_something(self):
        pc1 = np.array([
            [
                [1., 2., 3.],
                [2., 3., 4.],
                [20., 20., 20]
            ],
            [
                [1., 2., 3.],
                [2., 3., 4.],
                [10., 10., 20]
            ]
        ])
        pc2 = np.array([
            [
                [2., 2., 2.],
                [2., 15., 15.],
                [10., 1., 20]
            ],
            [
                [2., 2., 2.],
                [2., 1., 15.],
                [10., 10., 20]
            ]
        ])
        pc1 = torch.from_numpy(pc1).float()
        pc2 = torch.from_numpy(pc2).float()
        l1 = ChamferLoss()(pc1, pc2)
        l2 = CustomChamferDistance()(pc1, pc2)
        print(l1, l2)


if __name__ == '__main__':
    unittest.main()
