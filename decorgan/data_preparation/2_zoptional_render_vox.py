import numpy as np
import cv2
import os
import binvox_rw
import argparse

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


def run(in_name):
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


for f_or_d in os.listdir(target_dir):
    if os.path.isdir(os.path.join(target_dir, f_or_d)):
        if f_or_d in obj_names:
            for f_or_d_2 in os.listdir(os.path.join(target_dir, f_or_d)):
                if os.path.isfile(os.path.join(target_dir, f_or_d, f_or_d_2)) and f_or_d_2 == "model.binvox":
                    name_in = os.path.join(target_dir, f_or_d, f_or_d_2)
                    run(name_in)
                else:
                    name_in = os.path.join(target_dir, f_or_d, f_or_d_2, "model.binvox")
                    if not os.path.exists(name_in):
                        print(f"couldnt run for non existing {name_in}")
                        continue
                    run(name_in)
                if os.path.isfile(os.path.join(target_dir, f_or_d, f_or_d_2)) and f_or_d_2 == "model_filled.binvox":
                    name_in = os.path.join(target_dir, f_or_d, f_or_d_2)
                    run(name_in)
                else:
                    name_in = os.path.join(target_dir, f_or_d, f_or_d_2, "model_filled.binvox")
                    if not os.path.exists(name_in):
                        print(f"couldnt run for non existing {name_in}")
                        continue
                    run(name_in)
