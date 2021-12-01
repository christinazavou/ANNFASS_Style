from unittest import TestCase, skip

import numpy as np

from mesh_utils import write_ply_v_f, ObjMeshWithComponentsAndMaterials


class Test(TestCase):
    @skip("skipping test_write_ply_v_f")
    def test_write_ply_v_f(self):
        v = np.random.random((100, 3))
        f = np.random.randint(0, 100, (50, 3))
        write_ply_v_f(v, f, "test.ply")

    def test_objs(self):
        o1 = ObjMeshWithComponentsAndMaterials("/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/normalizedObj/02_Panagia_Chrysaliniotissa/02_Panagia_Chrysaliniotissa.obj")
        o2 = ObjMeshWithComponentsAndMaterials("/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/normalizedObj/02_Panagia_Chrysaliniotissa/02_Panagia_Chrysaliniotissa_refinedTextures.obj")
        assert len(o1.texture_coords) != len(o2.texture_coords)
        assert len(o1.vertex_coords) == len(o2.vertex_coords)
        assert len(o1.components) == len(o2.components)
        assert len(o1.faces) != len(o2.faces)

