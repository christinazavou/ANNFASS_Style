import numpy as np
import os
import cutils
import argparse
# import cv2
# import mcubes
import sys
sys.path.extend(os.path.dirname(os.path.abspath(__file__)))
from common_utils import binvox_rw_customized

parser = argparse.ArgumentParser()
parser.add_argument("class_id", type=str, help="shapenet category id")
parser.add_argument("target_dir", type=str, default="./preprocessed")
parser.add_argument("share_id", type=int, help="id of the share [0]")
parser.add_argument("share_total", type=int, help="total num of shares [1]")
FLAGS = parser.parse_args()

class_id = FLAGS.class_id
target_dir = os.path.join(FLAGS.target_dir, class_id)
if not os.path.exists(target_dir):
    print("ERROR: this dir does not exist: "+target_dir)
    exit()

share_id = FLAGS.share_id
share_total = FLAGS.share_total

obj_names = os.listdir(target_dir)
obj_names = sorted(obj_names)

start = int(share_id*len(obj_names)/share_total)
end = int((share_id+1)*len(obj_names)/share_total)
obj_names = obj_names[start:end]


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


def run(in_name, out_name):

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

        # voxel_model_file = open(out_name, 'rb')
        # vox_model = binvox_rw_customized.read_as_3d_array(voxel_model_file)
        # batch_voxels = vox_model.data.astype(np.uint8)
        # vertices, triangles = mcubes.marching_cubes(batch_voxels, 0.5)
        # write_ply_triangle(target_dir + obj_names[i] + "/df17_vox.ply", vertices, triangles)
        #
        # for j in range(17):
        #     img = (rendering[:,:,j]<2**16).astype(np.uint8)*255
        #     cv2.imwrite(target_dir + obj_names[i] + "/df17_vox_"+str(j)+".png", img)


for f_or_d in os.listdir(target_dir):
    if os.path.isdir(os.path.join(target_dir, f_or_d)):
        if f_or_d in obj_names:
            for f_or_d_2 in os.listdir(os.path.join(target_dir, f_or_d)):
                if os.path.isfile(os.path.join(target_dir, f_or_d, f_or_d_2)) and f_or_d_2 == "model.binvox":
                    name_in = os.path.join(target_dir, f_or_d, f_or_d_2)
                    name_out = os.path.join(target_dir, f_or_d, "model_depth_fusion_17.binvox")
                    run(name_in, name_out)
                else:
                    name_in = os.path.join(target_dir, f_or_d, f_or_d_2, "model.binvox")
                    if not os.path.exists(name_in):
                        print(f"couldnt run for non existing {name_in}")
                        continue
                    name_out = os.path.join(target_dir, f_or_d, f_or_d_2, "model_depth_fusion_17.binvox")
                    run(name_in, name_out)
