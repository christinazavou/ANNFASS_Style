import argparse
import os
import shlex
import subprocess
import sys
import time

import numpy as np
from scipy import spatial

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from common.utils import str2bool, parse_buildings_csv
from common.mesh_utils import read_pts
from common.multiprocessing_utils import run_function_in_parallel


def create_ridge_valley_files(ridge_file, valley_file, model_id, obj_filename, **kwargs):
    if not os.path.exists(ridge_file) or not os.path.exists(valley_file) or kwargs['override']:
        try:
            rnvCMD = kwargs['rnv'] + ' "' + obj_filename + '"'
            print(rnvCMD)
            proc = subprocess.Popen(shlex.split(rnvCMD), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()
        except Exception as e:
            raise Exception("No generated samples for {}.".format(model_id), e)
        assert os.path.isfile(ridge_file) and os.path.isfile(valley_file), "RNV couldn't run correctly for {}.".format(
            model_id)


def create_ridge_valley_label(models, process_id, **kwargs):
    """
    :param models list containing modelId, modelFile
    :param process_id id of the process since parallel processes can be called
    """

    t_start_proc = time.time()

    if kwargs['debug']:
        print("Starting process {process_id:d}...".format(process_id=process_id))

    startIdx = 0
    for i in range(startIdx, len(models)):
        model_id, obj_filename, pts_filein, txt_fileout = models[i]

        if kwargs['debug']:
            print("Start of {} (process: {})".format(model_id, process_id))

        ridge_file = obj_filename + ".ridge.txt"
        valley_file = obj_filename + ".valley.txt"
        create_ridge_valley_files(ridge_file, valley_file, model_id, obj_filename, **kwargs)

        if kwargs['debug']:
            print("Processing the results from RidgeValleyExporter for {}".format(model_id))

        sampled_xyz, sampled_normals = read_pts(pts_filein)
        avg_min_distance = get_avg_min_distance_of_sampled_points(sampled_xyz)
        ridges_xyz = get_ridge_or_valley_vertices(ridge_file, avg_min_distance)
        valleys_xyz = get_ridge_or_valley_vertices(valley_file, avg_min_distance)

        rnv_labels = get_rnv_labels(sampled_xyz, avg_min_distance, ridges_xyz, valleys_xyz)
        write_rnv_labels(rnv_labels, txt_fileout)

        if kwargs['remove']:
            os.remove(ridge_file)
            os.remove(valley_file)

        if kwargs['debug']:
            print("Preprocessing model {model_id:s} was successful".format(model_id=model_id))

        print("Process {}: Processed files ({}/{})".format(process_id, i + 1, len(models)))

    elapsed_time = time.time() - t_start_proc
    print("Terminating process {process_id:d}. Time passed: {hours:d}:{minutes:d}:{seconds:d}"
          .format(process_id=process_id,
                  hours=int((elapsed_time / 60 ** 2) % (60 ** 2)),
                  minutes=int((elapsed_time / 60) % (60)),
                  seconds=int(elapsed_time % 60)))


def write_ridge_valley_labels(sampled_xyz, pts_r_indices, pts_v_indices, txt_fileout):
    with open(txt_fileout, "w") as fout:
        for idx in range(len(sampled_xyz)):
            if idx in pts_r_indices:
                if idx in pts_v_indices:
                    fout.write("2\n")
                else:
                    fout.write("1\n")
            elif idx in pts_v_indices:
                fout.write("-1\n")
            else:
                fout.write("0\n")


def write_rnv_labels(labels, txt_fileout):
    with open(txt_fileout, "w") as fout:
        for lbl in labels:
            fout.write("{}\n".format(lbl))


def get_avg_min_distance_of_sampled_points(sampled_xyz):
    tree = spatial.cKDTree(sampled_xyz, leafsize=500)
    min_distances = []
    for xyz in sampled_xyz:
        distances, indices = tree.query(xyz, k=2)
        assert distances[0] == 0
        min_distances.append(distances[1])
    return np.mean(min_distances)


def get_ridge_or_valley_vertices(ridge_or_valley_txt, avg_min_distance):
    ridge_or_valley_vertices = np.loadtxt(ridge_or_valley_txt)
    assert ridge_or_valley_vertices.shape[0] % 2 == 0
    ridge_or_valley_edges = np.reshape(ridge_or_valley_vertices, (int(ridge_or_valley_vertices.shape[0]/2), 2, 3))
    sampled_vertices_on_edges = []
    for edge in ridge_or_valley_edges:
        v0 = edge[0]
        v1 = edge[1]
        line = v1 - v0
        if np.linalg.norm(line) / avg_min_distance >= 2:
            new_samples_cnt = np.round(np.linalg.norm(line)/avg_min_distance).astype(int) + 1
            distances = np.linspace(0, 1, new_samples_cnt)
            distances = np.expand_dims(distances[1:-1], 1)
            new_vertices = v0 + distances @ np.expand_dims(line, 0)
            sampled_vertices_on_edges.append(new_vertices)
    if len(sampled_vertices_on_edges) > 0 :
        sampled_vertices_on_edges = np.vstack(sampled_vertices_on_edges)
        return np.vstack([ridge_or_valley_vertices, sampled_vertices_on_edges])
    return ridge_or_valley_vertices


def get_rnv_labels(sampled_xyz, avg_min_distance, ridges_xyz, valleys_xyz):
    factor = 1.2
    if len(ridges_xyz) > 0:
        ridges_tree = spatial.cKDTree(ridges_xyz, leafsize=500)
    if len(valleys_xyz) > 0:
        valleys_tree = spatial.cKDTree(valleys_xyz, leafsize=500)
    labels = []
    for point in sampled_xyz:
        lbl = 0
        distance_r = 1e9
        if len(ridges_xyz) > 0:
            distance_r, index = ridges_tree.query(point, k=1)
            if distance_r <= avg_min_distance*factor:
                lbl = 1
        if len(valleys_xyz) > 0:
            distance_v, index = valleys_tree.query(point, k=1)
            if distance_v <= avg_min_distance*factor:
                if distance_v < distance_r:
                    lbl = -1
        labels.append(lbl)
    return labels


def get_sampled_point_indices_closest_to_rnv_points(rnv_points, sampled_points):
    samples_tree = spatial.cKDTree(sampled_points, leafsize=500)
    k = 3
    dist, k_nn = samples_tree.query(rnv_points, k=k)
    k_nn = np.unique(k_nn.reshape((-1)))
    return k_nn


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", type=int, default=8, help="Number of processes to use [default: 1]")
    parser.add_argument("--buildings_csv", required=True, help="Buildings to process", type=str)
    parser.add_argument("--debug", default=False, type=str2bool)
    parser.add_argument("--rnv", type=str, required=True, help="executable file")
    parser.add_argument("--obj_dir", type=str, required=True, help="Data with normalized obj files directory")
    parser.add_argument("--pts_dir", type=str, required=True, help="Data with normalized obj files directory")
    parser.add_argument("--override", type=str2bool, default=True, help="Whether to override tmp file")
    parser.add_argument("--remove", type=str2bool, default=True, help="Whether to remove tmp file")
    ARGS = parser.parse_args()
    print("ARGS: {}".format(ARGS))

    OUTPUT_DIR = ARGS.pts_dir.replace("point_cloud", "ridge_or_valley")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Starting ridges and valleys creation of points.")
    buildings = parse_buildings_csv(ARGS.buildings_csv)

    model_list = []
    for building in buildings:

        pts_file = os.path.join(ARGS.pts_dir, "{}.pts".format(building))
        out_file = os.path.join(OUTPUT_DIR, "{}.txt".format(building))

        if not os.path.exists(pts_file):
            continue

        if os.path.exists(out_file):
            continue  # dont override

        obj_file = os.path.join(ARGS.obj_dir, building, "{}.obj".format(building))

        if os.path.exists(obj_file):
            model_list.append([building, obj_file, pts_file, out_file])

    print("models to process: {}\n".format(model_list))

    # Preprocess models
    t1 = time.time()
    run_function_in_parallel(create_ridge_valley_label, ARGS.num_processes, model_list,
                             debug=ARGS.debug, rnv=ARGS.rnv, override=ARGS.override, remove=ARGS.remove)

    total_time = time.time() - t1
    print("Finished all processes. Time passed: {hours:d}:{minutes:d}:{seconds:d}"
          .format(hours=int((total_time / 60 ** 2) % (60 ** 2)),
                  minutes=int((total_time / 60) % 60),
                  seconds=int(total_time % 60)))

    # for debugging, comment out the above and use this:
    # create_ridge_valley_label(model_list[:2], 0,
    #                           debug=ARGS.debug, rnv=ARGS.rnv, override=ARGS.override, remove=ARGS.remove)
