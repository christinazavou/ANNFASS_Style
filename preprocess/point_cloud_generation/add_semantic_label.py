import argparse
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from common.utils import parse_buildings_csv, BUILDNET_ELEMENTS
from common.mesh_utils_io import SampledPoints, ObjMeshComponentsReference, write_ply_with_normals_and_others, \
    SampledPointsWithColor
from common.multiprocessing_utils import run_function_in_parallel


def append_label_to_new_ply(models, process_id, **kwargs):
    """
    :param models list containing modelId, modelFile
    :param process_id id of the process since parallel processes can be called
    """

    t_start_proc = time.time()

    print("Starting process {process_id:d}...".format(process_id=process_id))

    startIdx = 0
    for i in range(startIdx, len(models)):
        model_id, obj_filename, ply_filein, face_filein, pts_filein, ply_fileout = models[i]

        sampled_points = SampledPoints()(pts_filein, face_filein)
        sampled_points_with_colour = SampledPointsWithColor()(ply_filein, face_filein)

        obj = ObjMeshComponentsReference(obj_filename)

        write_with_part_segmentation_labels(sampled_points, sampled_points_with_colour, obj, ply_fileout)

        print("Process {}: Processed files ({}/{})".format(process_id, i + 1, len(models)))

    elapsed_time = time.time() - t_start_proc
    print("Terminating process {process_id:d}. Time passed: {hours:d}:{minutes:d}:{seconds:d}"
          .format(process_id=process_id,
                  hours=int((elapsed_time / 60 ** 2) % (60 ** 2)),
                  minutes=int((elapsed_time / 60) % (60)),
                  seconds=int(elapsed_time % 60)))


def write_with_part_segmentation_labels(sampled_points, sampled_points_with_colour, obj, ply_fileout):
    coords, normals, rgbs, labels = [], [], [], []
    for point, point_with_color in zip(sampled_points, sampled_points_with_colour):
        face = obj.faces[point.face_idx]
        component = face.component
        label = [key for key, value in BUILDNET_ELEMENTS.items() if value.lower() in component.lower()]
        if len(label) == 0:
            label = ['0']  # undetermined fixme: OK ?
        labels.append(int(label[0]))
        coords.append(point.coords)
        normals.append(point.normals)
        rgbs.append(point_with_color.color_interp)
    rgbs_and_labels = np.append(rgbs, np.array(labels).reshape((len(labels), 1)), 1).astype(np.int)
    write_ply_with_normals_and_others(ply_fileout, np.vstack(coords), np.vstack(normals), rgbs_and_labels,
                                      ['red', 'green', 'blue', 'label'],
                                      ['uchar', 'uchar', 'uchar', 'int'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", type=int, default=1, help="Number of processes to use [default: 1]")
    parser.add_argument("--buildings_csv", required=True, help="Buildings to process", type=str)
    parser.add_argument("--obj_dir", type=str, required=True)
    parser.add_argument("--ply_dir", type=str, required=True)
    parser.add_argument("--pts_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    ARGS = parser.parse_args()
    print("ARGS: {}".format(ARGS))

    buildings = parse_buildings_csv(ARGS.buildings_csv)
    os.makedirs(ARGS.out_dir, exist_ok=True)

    model_list = []
    for building in buildings:

        ply_file = os.path.join(ARGS.ply_dir, "{}.ply".format(building))
        face_file = os.path.join(ARGS.pts_dir.replace("point_cloud", "faces"), "{}.txt".format(building))
        pts_file = os.path.join(ARGS.pts_dir, "{}.pts".format(building))
        out_file = os.path.join(ARGS.out_dir, "{}.ply".format(building))

        if not os.path.exists(ply_file) or not os.path.exists(face_file) or not os.path.exists(pts_file):
            continue

        if os.path.exists(out_file):
            continue  # dont override

        obj_file = os.path.join(ARGS.obj_dir, building, "{}.obj".format(building))

        if os.path.exists(obj_file):
            model_list.append([building, obj_file, ply_file, face_file, pts_file, out_file])

    print("models to process: {}\n".format(model_list))

    # Preprocess models
    t1 = time.time()
    run_function_in_parallel(append_label_to_new_ply, ARGS.num_processes, model_list,)
    # for debugging, comment out the above and use this:
    # append_label_to_new_ply(model_list[:2], 0)

    total_time = time.time() - t1
    print("Finished all processes. Time passed: {hours:d}:{minutes:d}:{seconds:d}"
          .format(hours=int((total_time / 60 ** 2) % (60 ** 2)),
                  minutes=int((total_time / 60) % 60),
                  seconds=int(total_time % 60)))
