import argparse
import math
import os
from os.path import join

import mathutils
import numpy as np

_THRESHOLD_TOL_32 = 2.0 * np.finfo(np.float32).eps
_THRESHOLD_TOL_64 = 2.0 * np.finfo(np.float32).eps


def load_obj(filename, normalize=True):
    fin = open(filename, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []
    faces = []
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            if len(line.split()) != 4:
                continue
            if "//" in line.split()[1]:
                faces.append(np.int32([item.split('//')[0] for item in line.split()[1:4]]))
            elif "/" in line.split()[1]:
                faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))
            else:
                faces.append(np.int32(line.split()[1:4]))

    if len(faces) > 0:
        f = np.vstack(faces)
        if np.amin(faces) == 1:
            f -= 1
    else:
        print(filename + " does not have any faces")
        f = np.array([])

    v = np.vstack(vertices)

    if normalize:
        # normalize diagonal=1
        x_max = np.max(v[:, 0])
        y_max = np.max(v[:, 1])
        z_max = np.max(v[:, 2])
        x_min = np.min(v[:, 0])
        y_min = np.min(v[:, 1])
        z_min = np.min(v[:, 2])
        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2
        z_mid = (z_max + z_min) / 2
        x_scale = x_max - x_min
        y_scale = y_max - y_min
        z_scale = z_max - z_min
        scale = np.sqrt(x_scale * x_scale + y_scale * y_scale + z_scale * z_scale)

        v[:, 0] = (v[:, 0] - x_mid) / scale
        v[:, 1] = (v[:, 1] - y_mid) / scale
        v[:, 2] = (v[:, 2] - z_mid) / scale

    return v, f


def write_simple_obj(out_fn, v, f):
    if np.amin(f) == 0:
        f += 1
    with open(out_fn, "w") as fout:
        for vertex in v:
            fout.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in f:
            fout.write(f"f {int(face[0])} {int(face[1])} {int(face[2])}\n")


def normalize_coords(coords, method="sphere"):
    """ Normalize coordinates """
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid

    if method.lower() == "sphere":
        radius = bounding_sphere_radius(coords=centered_coords)
    elif method.lower() == "box":
        radius = bounding_box_diagonal(coords=centered_coords)
    else:
        print("Unknown normalization method {}".format(method))
        exit(-1)

    radius = np.maximum(radius, _THRESHOLD_TOL_64 if radius.dtype == np.float64 else _THRESHOLD_TOL_32)

    return centered_coords / radius


def bounding_box_diagonal(coords):
    """ Return bounding box diagonal """
    bb_diagonal = np.array([np.amax(coords[:, 0]), np.amax(coords[:, 1]), np.amax(coords[:, 2])]) - \
                  np.array([np.amin(coords[:, 0]), np.amin(coords[:, 1]), np.amin(coords[:, 2])])
    bb_diagonal_length = np.sqrt(np.sum(bb_diagonal ** 2))

    return bb_diagonal_length


def bounding_sphere_radius(coords):
    """ Return bounding sphere radius """
    radius = np.max(np.sqrt(np.sum(coords ** 2, axis=1)))

    return radius


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--random_rot", type=str, default="True")
    parser.add_argument("--rotations", type=int, default=0, help="Default would be 0 for no rotations or 10 otherwise")
    parser.add_argument("--keep_original", type=str, default="True", help="Default would be 0 for no rotations or 10 otherwise")
    ARGS = parser.parse_args()
    print(ARGS)

    for model_dir in os.listdir(ARGS.inp_dir):
        inp_file = join(ARGS.inp_dir, model_dir, "model.obj")
        out_file = join(ARGS.out_dir, model_dir, "model.obj")

        os.makedirs(ARGS.out_dir, exist_ok=True)

        if eval(ARGS.keep_original) is False:
            if ARGS.rotations > 0:
                existing = [f for f in os.listdir(ARGS.out_dir) if f"{model_dir}_rot" in f]
                remaining = ARGS.rotations - len(existing)
                if remaining == 0:
                    continue

        vertices, faces = load_obj(inp_file, normalize=True)

        # vertices = normalize_coords(vertices)

        if eval(ARGS.keep_original) is True:
            os.makedirs(join(ARGS.out_dir, model_dir), exist_ok=True)
            write_simple_obj(out_file, vertices, faces)

        if ARGS.rotations > 0:
            existing = [f for f in os.listdir(ARGS.out_dir) if f"{model_dir}_rot" in f]
            remaining = ARGS.rotations - len(existing)

            if eval(ARGS.random_rot):
                for i in range(remaining):
                    angle = np.random.randint(0, 360)
                    while os.path.exists(join(ARGS.out_dir, f"{model_dir}_rot{angle}")):
                        angle = np.random.randint(0, 360)

                    rot = mathutils.Matrix.Rotation(math.radians(angle), 3, (0, 1, 0))
                    vertices_rot = np.matmul(vertices, rot)

                    os.makedirs(join(ARGS.out_dir, f"{model_dir}_rot{angle}"), exist_ok=True)
                    write_simple_obj(join(ARGS.out_dir, f"{model_dir}_rot{angle}", f"model.obj"), vertices_rot, faces)
            else:
                for i in range(1, ARGS.rotations):
                    angle = int((360 / ARGS.rotations) * i)
                    rot = mathutils.Matrix.Rotation(math.radians(angle), 3, (0, 1, 0))
                    vertices_rot = np.matmul(vertices, rot)

                    os.makedirs(join(ARGS.out_dir, f"{model_dir}_rot{angle}"), exist_ok=True)
                    write_simple_obj(join(ARGS.out_dir, f"{model_dir}_rot{angle}", f"model.obj"), vertices_rot, faces)
