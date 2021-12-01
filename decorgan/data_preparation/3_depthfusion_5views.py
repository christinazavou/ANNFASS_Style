import numpy as np
import cv2
import os
import mcubes
import cutils
import argparse
import sys
sys.path.extend(os.path.dirname(os.path.abspath(__file__)))
from common_utils import binvox_rw_customized

parser = argparse.ArgumentParser()
parser.add_argument("class_id", type=str, help="shapenet category id")
parser.add_argument("share_id", type=int, help="id of the share [0]")
parser.add_argument("share_total", type=int, help="total num of shares [1]")
parser.add_argument("target_dir", type=int, default="./preprocessed")
FLAGS = parser.parse_args()

class_id = FLAGS.class_id
target_dir = os.path.join(FLAGS.target_dir, class_id)
if not os.path.exists(target_dir):
    print("ERROR: this dir does not exist: "+target_dir)
    exit()

share_id = FLAGS.share_id
share_total = FLAGS.share_total


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


rendering = np.zeros([2560,2560,5], np.int32)
state_ctr = np.zeros([512*512*64,2], np.int32)


def run(file_in, file_out):
    if os.path.exists(file_out):
        return
    print(file_in)

    voxel_model_file = open(file_in, 'rb')
    vox_model = binvox_rw_customized.read_as_3d_array(voxel_model_file,fix_coords=False)

    batch_voxels = vox_model.data.astype(np.uint8)
    rendering[:] = 2**16
    cutils.depth_fusion_XZY_5views(batch_voxels,rendering,state_ctr)


    with open(file_out, 'wb') as fout:
        binvox_rw_customized.write(vox_model, fout, state_ctr)

    # voxel_model_file = open(out_name, 'rb')
    # vox_model = binvox_rw_customized.read_as_3d_array(voxel_model_file)
    # batch_voxels = vox_model.data.astype(np.uint8)
    # vertices, triangles = mcubes.marching_cubes(batch_voxels, 0.5)
    # write_ply_triangle(target_dir + obj_names[i] + "/df_vox.ply", vertices, triangles)
    #
    # for j in range(5):
    #     img = (rendering[:,:,j]<2**16).astype(np.uint8)*255
    #     cv2.imwrite(target_dir + obj_names[i] + "/df_vox_"+str(j)+".png", img)



f_or_d_names = os.listdir(target_dir)
f_or_d_names = sorted(f_or_d_names)

start = int(share_id*len(f_or_d_names)/share_total)
end = int((share_id+1)*len(f_or_d_names)/share_total)
f_or_d_names = f_or_d_names[start:end]

for f_or_d in f_or_d_names:
    if os.path.isdir(os.path.join(target_dir, f_or_d)):
        for f_or_d_2 in os.listdir(os.path.join(target_dir, f_or_d)):
            if os.path.isfile(os.path.join(target_dir, f_or_d, f_or_d_2)) and f_or_d_2 == "model.binvox":
                this_name = os.path.join(target_dir, f_or_d, f_or_d_2)
                out_name = os.path.join(target_dir, f_or_d, "model_depth_fusion.binvox")
                run(this_name, out_name)
            else:
                this_name = os.path.join(target_dir, f_or_d, f_or_d_2, "model.binvox")
                assert os.path.exists(this_name)
                out_name = os.path.join(target_dir, f_or_d, f_or_d_2, "model_depth_fusion.binvox")
                run(this_name, out_name)


