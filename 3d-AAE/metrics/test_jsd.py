from unittest import TestCase
from jsd import _unit_cube_grid_point_cloud, _entropy_of_occupancy_grid, jsd_between_point_cloud_sets
import numpy as np
import torch


class Test(TestCase):
    def test__unit_cube_grid_point_cloud(self):
        grid, spacing = _unit_cube_grid_point_cloud(8 ** 3)
        print(grid.reshape(-1, 3).shape)


class Test(TestCase):
    def test__entropy_of_occupancy_grid(self):
        point_clouds = np.random.random((4, 2048, 3))
        point_clouds = torch.tensor(point_clouds).cuda()
        res = _entropy_of_occupancy_grid(point_clouds, 2 ** 7)
        print(res)


class Test(TestCase):
    def test_js_divercence_between_pc(self):
        pc1 = np.random.random((4, 2048, 3))
        pc1 = torch.tensor(pc1)
        pc2 = np.random.random((4, 2048, 3))
        pc2 = torch.tensor(pc2)
        jsd = jsd_between_point_cloud_sets(pc2, pc2)
