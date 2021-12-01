import argparse
import copy
import json
import os
import sys

import numpy as np

from laplacian import mesh_taubin_smoothing

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.mesh_utils import ObjMeshWithComponents, ObjMeshWithAdjacency, MeshWithAdjacency, \
    get_component_mesh, write_ply_v_f, TriangleMesh, get_normalized_mesh
from common.utils import parse_buildings_csv, str2bool


def smooth_building_first(obj_file, out_dir, components, iterations=3, lamb=0.15, mu=0.16, debug=False):
    if os.path.exists(out_dir):
        return
    os.makedirs(out_dir)

    building_mesh = ObjMeshWithComponents(obj_file)
    smooth_building_mesh = copy.deepcopy(building_mesh)
    building_mesh_adg = ObjMeshWithAdjacency(obj_file)
    assert np.array_equal(building_mesh.vertex_coords, building_mesh_adg.vertices)

    # faces = np.array([f.v_indices for f in building_mesh.faces])
    # trmeshbefore = TriangleMesh(building_mesh_adg.vertices, faces)
    mesh_taubin_smoothing(building_mesh_adg, iterations=iterations, lamb=lamb, mu=mu)
    smooth_building_mesh.vertex_coords = building_mesh_adg.vertices
    # trmeshafter = TriangleMesh(building_mesh_adg.vertices, faces)
    # o3d.draw_geometries([trmeshbefore.translate((-0.6, 0., 0.)), trmeshafter.translate((+0.6, 0., 0.))])

    for component in components:
        smoothed_vertices, smoothed_faces = get_component_mesh(smooth_building_mesh, component)
        tri_mesh = TriangleMesh(smoothed_vertices, smoothed_faces)
        tri_mesh = get_normalized_mesh(tri_mesh)
        write_ply_v_f(tri_mesh.vertices, tri_mesh.triangles, os.path.join(out_dir, f"{component}.ply"))

        if debug:
            vertices, faces = get_component_mesh(building_mesh, component)
            tri_mesh_init = TriangleMesh(vertices, faces)
            tri_mesh_init = get_normalized_mesh(tri_mesh_init)
            write_ply_v_f(tri_mesh_init.vertices, tri_mesh_init.triangles, os.path.join(out_dir, f"{component}_init.ply"))
            # o3d.draw_geometries([tri_mesh_init.translate((-0.6, 0., 0.)), tri_mesh.translate((+0.6, 0., 0.))])


def smooth_component_direct(obj_file, out_dir, components, iterations=3, lamb=0.15, mu=0.16, debug=False):
    if os.path.exists(out_dir):
        return
    os.makedirs(out_dir)

    building_mesh = ObjMeshWithComponents(obj_file)
    building_mesh_adg = ObjMeshWithAdjacency(obj_file)
    assert np.array_equal(building_mesh.vertex_coords, building_mesh_adg.vertices)

    for component in components:
        vertices, faces, old_idx_new_idx = get_component_mesh(building_mesh, component, return_map=True)
        v_new_existing_neighbours = dict()
        for v_old_idx in old_idx_new_idx:
            v_new_idx = old_idx_new_idx[v_old_idx]
            v_old_neighbours = building_mesh_adg.vertices_neighbours[v_old_idx]
            v_new_neighbours = [old_idx_new_idx[v] for v in v_old_neighbours if v in old_idx_new_idx]
            v_new_existing_neighbours[v_new_idx] = v_new_neighbours

        mesh = MeshWithAdjacency()
        mesh.vertices = vertices
        mesh.faces = faces
        mesh.vertices_neighbours = v_new_existing_neighbours

        if debug:
            tri_mesh_init = TriangleMesh(mesh.vertices, mesh.faces)
            tri_mesh_init = get_normalized_mesh(tri_mesh_init)
            write_ply_v_f(tri_mesh_init.vertices, tri_mesh_init.triangles, os.path.join(out_dir, f"{component}_init.ply"))

        mesh_taubin_smoothing(mesh, iterations=iterations, lamb=lamb, mu=mu)
        tri_mesh = TriangleMesh(mesh.vertices, mesh.faces)
        tri_mesh = get_normalized_mesh(tri_mesh)
        # o3d.draw_geometries([tri_mesh_init.translate((-0.6, 0., 0.)), tri_mesh.translate((+0.6, 0., 0.))])
        write_ply_v_f(tri_mesh.vertices, tri_mesh.triangles, os.path.join(out_dir, f"{component}.ply"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data', type=str, required=True)
    parser.add_argument('--obj_dir', type=str, default="normalizedObj")
    parser.add_argument('--unique_dir', type=str, default="uncom")
    parser.add_argument('--out_dir', type=str, default="uncom_smoothed")
    parser.add_argument('--buildings_csv', type=str, default="buildings_religious.csv")
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--lamb', type=float, default=0.40)
    parser.add_argument('--mu', type=float, default=0.41)
    parser.add_argument('--debug', type=str2bool, default=False)
    args = parser.parse_args()

    unique_components_dir = os.path.join(args.root_data, args.unique_dir)
    buildings = parse_buildings_csv(os.path.join(args.root_data, args.buildings_csv))

    for building in buildings:
        duplicates_file = os.path.join(unique_components_dir, building, "duplicates.json")
        if not os.path.exists(duplicates_file):
            continue
        det_out_dir = args.out_dir + f"iter{args.iterations}lamb{args.lamb}mu{args.mu}"
        with open(duplicates_file, 'r') as fin:
            duplicates = json.load(fin)
            smooth_building_first(os.path.join(args.root_data, args.obj_dir, building, f"{building}.obj"),
                                  os.path.join(args.root_data, det_out_dir, building, "building_first"),
                                  duplicates.keys(),
                                  iterations=args.iterations, lamb=args.lamb, mu=args.mu, debug=args.debug)
            # smooth_component_direct(os.path.join(args.root_data, args.obj_dir, building, f"{building}.obj"),
            #                         os.path.join(args.root_data, det_out_dir, building, "component_direct"),
            #                         duplicates.keys(),
            #                         iterations=args.iterations, lamb=args.lamb, mu=args.mu, debug=args.debug)
