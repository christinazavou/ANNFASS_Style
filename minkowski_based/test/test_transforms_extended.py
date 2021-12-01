from unittest import TestCase
import numpy as np

from lib.transforms_extended import get_component_indices_matrix, get_average_per_component


class Test(TestCase):
    def test_componentfunctions(self):
        component_indices = [-1,-1,2,4,6,-1,-1,10]
        component_indices = np.array(component_indices, dtype=np.int32)

        component_names = ["b1_c1", "b1_c2", "b1_c3", "b2_c1", "b2_c2", "b3_c1", "b3_c2", "b2_c3", "b2_c4", "b2_c5"]

        cimat, cnames = get_component_indices_matrix(component_indices, component_names)
        print(cimat, cnames)

        point_data = np.array([[1,2], [2,3], [3,3], [4,4], [4,4], [5,5], [6,6], [3,3]])

        res = get_average_per_component(cimat, point_data)
        print(res)

