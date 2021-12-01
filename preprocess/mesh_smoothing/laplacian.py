import argparse
import os
import sys

import numpy as np
import robust_laplacian

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.mesh_utils import ObjMeshWithAdjacency, PlyWithAdjacency, MeshWithAdjacency
from common.utils import str2bool


def mesh_taubin_smoothing(mesh: MeshWithAdjacency, iterations=1, lamb=0.5, mu=0.51):
    vertices = mesh.vertices

    sorted_vert_indices = sorted(mesh.vertices_neighbours.keys())

    for index in range(iterations):
        # do a sparse dot product on the vertices
        laplacian_factors = []
        for v_idx in sorted_vert_indices:
            if mesh.vertices_neighbours[v_idx] != []:
                neigh_vertices = mesh.vertices[mesh.vertices_neighbours[v_idx]]
                avg_vertex = np.mean(neigh_vertices, 0)
                laplacian_factors.append(avg_vertex - mesh.vertices[v_idx])
            else:
                laplacian_factors.append(np.array([0., 0., 0.]))
        laplacian_factors = np.array(laplacian_factors)
        # alternate shrinkage and dilation
        if index % 2 == 0:
            vertices += lamb * laplacian_factors
        else:
            vertices -= mu * laplacian_factors
    mesh.vertices = vertices


def pointcloud_taubin_smoothing(points, points_neighbours, iterations=1, lamb=0.5, mu=0.51):
    sorted_vert_indices = sorted(points_neighbours.keys())

    for index in range(iterations):
        # do a sparse dot product on the vertices
        laplacian_factors = []
        for v_idx in sorted_vert_indices:
            if len(points_neighbours[v_idx]) > 0:
                neigh_vertices = points[points_neighbours[v_idx]]
                avg_vertex = np.mean(neigh_vertices, 0)
                laplacian_factors.append(avg_vertex - points[v_idx])
            else:
                laplacian_factors.append(np.array([0., 0., 0.]))
        laplacian_factors = np.array(laplacian_factors)
        # alternate shrinkage and dilation
        if index % 2 == 0:
            points += lamb * laplacian_factors
        else:
            points -= mu * laplacian_factors
    return points


def pointcloud_taubin_smoothing_with_laplacian(points, iterations=1, lamb=0.5, mu=0.51):

    L, M = robust_laplacian.point_cloud_laplacian(points)

    for index in range(iterations):
        # alternate shrinkage and dilation
        if index % 2 == 0:
            points += lamb * L.dot(points)
        else:
            points -= mu * L.dot(points)
    return points


def taubin_custom_obj(obj_file_in, obj_file_out):
    obj = ObjMeshWithAdjacency(obj_file_in)
    mesh_taubin_smoothing(obj, iterations=3, lamb=0.15, mu=0.16)
    with open(obj_file_in, "r") as fin:
        lines = fin.readlines()
    v_idx = -1

    with open(obj_file_out, "w") as fout:
        for line in lines:
            if line.startswith("v "):
                v_idx += 1
                fout.write(f"v {obj.vertices[v_idx][0]} {obj.vertices[v_idx][1]} {obj.vertices[v_idx][2]}"+"\n")
            elif line.startswith("vn "):
                pass
            elif line.startswith("vt "):
                pass
            else:
                fout.write(line)


def taubin_custom_ply(ply_file_in, ply_file_out):
    ply = PlyWithAdjacency(ply_file_in)
    mesh_taubin_smoothing(ply, iterations=3, lamb=0.15, mu=0.16)

    with open(ply_file_out, "w") as fout:
        header = 'ply\n' \
                 'format ascii 1.0\n' \
                 'element vertex ' + str(len(ply.vertices)) + '\n' \
                 'property float x\n' \
                 'property float y\n' \
                 'property float z\n' \
                 'element face ' + str(len(ply.faces)) + '\n' \
                 'property list uchar uint vertex_indices\n' \
                 'end_header\n'
        fout.write(header)
        for v in ply.vertices:
            fout.write(f"{v[0]} {v[1]} {v[2]}"+"\n")
        for f in ply.faces:
            fout.write("3 "+" ".join(f.astype(str))+"\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--debug', type=str2bool, default=False)
    args = parser.parse_args()

    with open(args.file, "r") as fin:
        paths_in = fin.readlines()
        for path in paths_in:
            taubin_custom_obj(path, f"{args.out_dir}/COMMERCIALfactory_mesh1454/COMMERCIALfactory_mesh1454.obj")
