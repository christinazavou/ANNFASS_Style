import numpy as np
from mathutils.bvhtree import BVHTree


class RayCastChecker:
    #
    def __init__(self, bm):
        self.bm_tree = BVHTree.FromBMesh(bm)
    #
    def shot_vertex(self, ray_origin, ray_dir, expected_vertex, logger=None):
        vertex_hit, normal_hit, index_hit, distance_hit = self.bm_tree.ray_cast(ray_origin, ray_dir)
        if vertex_hit == None:
            if logger:
                logger.warning("HUH??")
            return False
        hit = np.round(np.array(vertex_hit), 6)
        expected = np.round(np.array(expected_vertex), 6)
        if np.array_equal(hit, expected):
            return True
        else:
            if logger:
                logger.debug("vertex_hit: {}, expected_vertex: {}".format(vertex_hit, expected_vertex))
                logger.debug("hit: {}, expected: {}".format(hit, expected))
            return False
    #
    def hits_mesh_in_all_directions(self, point, directions):
        for direction in directions:
            vertex_hit, normal_hit, index_hit, distance_hit = self.bm_tree.ray_cast(point, direction)
            if vertex_hit == None:
                return False
        return True
