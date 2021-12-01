import os
import sys

import numpy as np
import argparse
import cv2
import mcubes
import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from common_utils import binvox_rw_customized
from utils.pytorch3d_vis import CustomMultiViewMeshRenderer
from common_utils.utils import normalize_vertices
from common_utils import binvox_rw
from data_preparation import cutils


BINVOX_COMMAND= "../binvox"


def load_obj(dire):
    fin = open(dire, 'r', encoding='latin1')
    lines = fin.readlines()
    fin.close()

    vertices = []
    triangles = []

    for i in range(len(lines)):
        line = lines[i].split()
        if len(line) == 0:
            continue
        if line[0] == 'v':
            x = float(line[1])
            y = float(line[2])
            z = float(line[3])
            vertices.append([x, y, z])
        if line[0] == 'f':
            assert len(line) == 4
            if "//" in line[1]:
                x = int(line[1].split("//")[0])
                y = int(line[2].split("//")[0])
                z = int(line[3].split("//")[0])
            else:
                x = int(line[1].split("/")[0])
                y = int(line[2].split("/")[0])
                z = int(line[3].split("/")[0])

            triangles.append([x - 1, y - 1, z - 1])

    vertices = np.array(vertices, np.float32)
    triangles = np.array(triangles, np.int32)

    # remove isolated points
    vertices_mapping = np.full([len(vertices)], -1, np.int32)
    for i in range(len(triangles)):
        for j in range(3):
            vertices_mapping[triangles[i, j]] = 1
    counter = 0
    for i in range(len(vertices)):
        if vertices_mapping[i] > 0:
            vertices_mapping[i] = counter
            counter += 1
    vertices = vertices[vertices_mapping >= 0]
    triangles = vertices_mapping[triangles]

    vertices = normalize_vertices(vertices)
    return vertices, triangles


def write_obj(dire, vertices, triangles):
    fout = open(dire, 'w')
    for ii in range(len(vertices)):
        fout.write("v " + str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + "\n")
    for ii in range(len(triangles)):
        fout.write(
            "f " + str(triangles[ii, 0] + 1) + " " + str(triangles[ii, 1] + 1) + " " + str(triangles[ii, 2] + 1) + "\n")
    fout.close()


def run_simplify(in_file, out_file):
    if os.path.exists(out_file):
        return
    print(in_file)
    try:
        v, t = load_obj(in_file)
    except:
        print(f"couldn't run for {in_file}")
        return
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    write_obj(out_file, v, t)


import subprocess
import shlex
def run_voxelize(in_file, out_file, binv_cmd=BINVOX_COMMAND, vox_size=512):
    if os.path.exists(out_file):
        return
    print(in_file)

    maxx = 0.5
    maxy = 0.5
    maxz = 0.5
    minx = -0.5
    miny = -0.5
    minz = -0.5

    command = binv_cmd+" -bb "+str(minx)+" "+str(miny)+" "+str(minz)+" "+str(maxx)+" "+str(maxy)+" "+str(maxz)+" "\
              +" -d "+str(vox_size)+" -e "+in_file
    print(command)
    proc = subprocess.Popen(shlex.split(command), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if err != b"":
        print(f"errors: {err}")


def write_ply_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face "+str(len(triangles))+"\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
    fout.close()


def run_step3(in_name, out_name):

    rendering = np.zeros([2560, 2560, 17], np.int32)
    state_ctr = np.zeros([512 * 512 * 64, 2], np.int32)

    if not os.path.exists(out_name):
        print(in_name)

        voxel_model_file = open(in_name, 'rb')
        vox_model = binvox_rw_customized.read_as_3d_array(voxel_model_file, fix_coords=False)

        batch_voxels = vox_model.data.astype(np.uint8)
        rendering[:] = 2**16
        cutils.depth_fusion_XZY(batch_voxels, rendering, state_ctr)

        with open(out_name, 'wb') as fout:
            binvox_rw_customized.write(vox_model, fout, state_ctr)


def run_floodfill(in_name, out_name):
    queue = np.zeros([512*512*64,3], np.int32)
    state_ctr = np.zeros([512*512*64,2], np.int32)

    if not os.path.exists(out_name):
        print(in_name)

        voxel_model_file = open(in_name, 'rb')
        vox_model = binvox_rw_customized.read_as_3d_array(voxel_model_file,fix_coords=False)

        batch_voxels = vox_model.data.astype(np.uint8)+1
        cutils.floodfill(batch_voxels,queue,state_ctr)

        with open(out_name, 'wb') as fout:
            binvox_rw_customized.write(vox_model, fout, state_ctr)

        # if you want to debug
        # voxel_model_file = open(out_name, 'rb')
        # vox_model = binvox_rw_customized.read_as_3d_array(voxel_model_file)
        # batch_voxels = vox_model.data.astype(np.uint8)
        # vertices, triangles = mcubes.marching_cubes(batch_voxels, 0.5)
        # vertices = normalize_vertices(vertices)  # if you want it in center
        # write_ply_triangle(out_name.replace(".binvox", ".ply"), vertices, triangles)


def run_rendering(in_name):
    voxel_model_file = open(in_name, 'rb')
    batch_voxels = binvox_rw.read_as_3d_array(voxel_model_file).data.astype(np.uint8)
    voxel_model_file.close()

    out = np.zeros([512 * 2, 512 * 4], np.uint8)

    tmp = batch_voxels
    mask = np.amax(tmp, axis=0).astype(np.int32)
    depth = np.argmax(tmp, axis=0)
    depth = 230 + np.clip(np.min(depth + (1 - mask) * 512) - depth, -180, 0)
    depth = depth * mask
    out[512 * 0:512 * 1, 512 * 0:512 * 1] = depth[::-1, :]

    mask = np.amax(tmp, axis=1).astype(np.int32)
    depth = np.argmax(tmp, axis=1)
    depth = 230 + np.clip(np.min(depth + (1 - mask) * 512) - depth, -180, 0)
    depth = depth * mask
    out[512 * 0:512 * 1, 512 * 1:512 * 2] = depth

    mask = np.amax(tmp, axis=2).astype(np.int32)
    depth = np.argmax(tmp, axis=2)
    depth = 230 + np.clip(np.min(depth + (1 - mask) * 512) - depth, -180, 0)
    depth = depth * mask
    out[512 * 0:512 * 1, 512 * 2:512 * 3] = np.transpose(depth)[::-1, ::-1]

    tmp = batch_voxels[::-1, :, :]
    mask = np.amax(tmp, axis=0).astype(np.int32)
    depth = np.argmax(tmp, axis=0)
    depth = 230 + np.clip(np.min(depth + (1 - mask) * 512) - depth, -180, 0)
    depth = depth * mask
    out[512 * 1:512 * 2, 512 * 0:512 * 1] = depth[::-1, ::-1]
    redisual = np.clip(np.abs(mask[:, :] - mask[:, ::-1]) * 256, 0, 255)
    out[512 * 0:512 * 1, 512 * 3:512 * 4] = redisual[::-1, ::-1]

    tmp = batch_voxels[:, ::-1, :]
    mask = np.amax(tmp, axis=1).astype(np.int32)
    depth = np.argmax(tmp, axis=1)
    depth = 230 + np.clip(np.min(depth + (1 - mask) * 512) - depth, -180, 0)
    depth = depth * mask
    out[512 * 1:512 * 2, 512 * 1:512 * 2] = depth[:, ::-1]
    redisual = np.clip(np.abs(mask[:, :] - mask[:, ::-1]) * 256, 0, 255)
    out[512 * 1:512 * 2, 512 * 3:512 * 4] = redisual[:, ::-1]

    tmp = batch_voxels[:, :, ::-1]
    mask = np.amax(tmp, axis=2).astype(np.int32)
    depth = np.argmax(tmp, axis=2)
    depth = 230 + np.clip(np.min(depth + (1 - mask) * 512) - depth, -180, 0)
    depth = depth * mask
    out[512 * 1:512 * 2, 512 * 2:512 * 3] = np.transpose(depth)[::-1, :]

    cv2.imwrite(in_name.replace(".binvox", ".png"), out)


def run_my_rendering(in_name):
    cmvr = CustomMultiViewMeshRenderer(torch.device('cuda:0'))

    if in_name.endswith(".obj"):
        img = cmvr(in_name, batch_size=10, show=False)
        cv2.imwrite(in_name.replace(".obj", "_mesh_mymultiview.png"), img)
    else:
        voxel_model_file = open(in_name, 'rb')
        batch_voxels = binvox_rw.read_as_3d_array(voxel_model_file).data.astype(np.uint8)
        voxel_model_file.close()

        vertices, triangles = mcubes.marching_cubes(batch_voxels, 0.5)
        vertices = normalize_vertices(vertices)
        img = cmvr(verts=vertices, triangles=triangles.astype(int), batch_size=10, show=False)
        cv2.imwrite(in_name.replace(".binvox", "_mymultiview.png"), img)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("class_id", type=str, )
    parser.add_argument("source_root", type=str, )
    parser.add_argument("target_dir", type=str, )
    parser.add_argument("share_id", type=int, )
    parser.add_argument("share_total", type=int, )
    FLAGS = parser.parse_args()

    class_id = FLAGS.class_id
    source_root = os.path.join(FLAGS.source_root, class_id)
    assert os.path.exists(source_root), f"{source_root} doesnt exist"

    target_dir = os.path.join(FLAGS.target_dir, class_id)
    os.makedirs(target_dir, exist_ok=True)

    share_id = FLAGS.share_id
    share_total = FLAGS.share_total

    model_dirs = os.listdir(source_root)
    model_dirs = sorted(model_dirs)

    start = int(share_id * len(model_dirs) / share_total)
    end = int((share_id + 1) * len(model_dirs) / share_total)
    model_dirs = model_dirs[start:end]

    idx_processed = 0
    for model_dir in model_dirs:
        if os.path.isdir(os.path.join(source_root, model_dir)):
            idx_processed += 1
            if idx_processed % 50 == 0:
                print(f"idx_processed {idx_processed}")

            obj_name = os.path.join(source_root, model_dir, "model.obj")
            assert os.path.exists(obj_name)

            model_name = os.path.join(target_dir, model_dir, "model.obj")
            if not os.path.exists(model_name):
                run_simplify(obj_name, model_name)
            binvox_name = os.path.join(target_dir, model_dir, "model.binvox")
            if not os.path.exists(binvox_name):
                run_voxelize(model_name, binvox_name)
            floodfill_name = os.path.join(target_dir, model_dir, "model_filled.binvox")
            if not os.path.exists(floodfill_name):
                run_floodfill(binvox_name, floodfill_name)
            # if not os.path.exists(floodfill_name.replace(".binvox", ".png")):
            #     run_rendering(floodfill_name)
