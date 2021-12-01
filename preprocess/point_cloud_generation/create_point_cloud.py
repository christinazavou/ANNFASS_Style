import argparse
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from preprocess.mesh_sampling.geometry_utils import ShapeFeatureExporter
from common.utils import str2bool
from common.multiprocessing_utils import log_process_time


def create_point_clouds(models, process_id, **kwargs):
    """
    :param models list containing modelId, modelFile
    :param process_id id of the process since parallel processes can be called
    """

    t_start_proc = time.time()

    debug = kwargs['debug']
    export_curvatures = kwargs['export_curvatures']
    export_pca = kwargs['export_pca']
    sfe = kwargs['sfe']
    override = kwargs['override']
    remove = kwargs['remove']
    points_dir = kwargs['points_dir']
    faces_dir = kwargs['faces_dir']

    if debug:
        print("Starting point cloud creation process {process_id:d}...".format(process_id=process_id))

    mandatory_args = ["n_samples"]
    # Check arguments
    for arg in mandatory_args:
        if arg not in kwargs:
            raise Exception("ArgumentError: {arg:s} is missing".format(arg=arg))

    startIdx = 0
    for i in range(startIdx, len(models)):
        model = models[i]
        model_id = model[0]
        model_filename = model[1]

        if debug:
            print("Poisson face sampling of {} (process: {})".format(model_id, process_id))

        args = '--do-not-rescale-shape --export-point-samples --num-point-samples {}'.format(kwargs['n_samples'])
        if export_curvatures:
            args += " --export-curvatures"
        if export_pca:
            args += " --export-principal-directions"

        try:
            shape_feature_exporter_result = ShapeFeatureExporter(model_filename, args, sfe, override, remove)
        except Exception as e:
            print(f"Warning: ShapeFeatureExporterException for {model_id}: {e}")
            continue

        assert len(shape_feature_exporter_result) > 1, "SFE couldn't run correctly for {}.".format(model_id)

        if debug:
            print("Processing the results from ShapeFeatureExporter")

        face_samples = shape_feature_exporter_result[0]
        face_normals = shape_feature_exporter_result[1]
        face_indices = shape_feature_exporter_result[2]
        face_curvatures = shape_feature_exporter_result[3]
        face_pca = shape_feature_exporter_result[4]
        del shape_feature_exporter_result

        # Save poisson sampled point cloud
        point_output_path = os.path.join(points_dir, "{}.pts".format(model_id))
        if export_curvatures and export_pca:
            point_cloud = np.hstack((face_samples, face_normals, face_curvatures, face_pca))
        elif export_curvatures:
            point_cloud = np.hstack((face_samples, face_normals, face_curvatures))
        elif export_pca:
            point_cloud = np.hstack((face_samples, face_normals, face_pca))
        else:
            point_cloud = np.hstack((face_samples, face_normals))

        export_results(point_output_path, point_cloud)
        del point_cloud, face_samples, face_normals, face_curvatures, face_pca

        # Save poisson samples face indices
        face_idx_output_path = os.path.join(faces_dir, "{}.txt".format(model_id))
        export_results(face_idx_output_path, np.expand_dims(face_indices, axis=1))
        del face_indices

        if debug:
            print("Preprocessing model {model_id:s} was successful".format(model_id=model_id))

        print("Process {}: Processed files ({}/{})".format(process_id, i + 1, len(models)))

    log_process_time(process_id, t_start_proc)


def export_results(output_path, data):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not os.path.exists(output_path):
        open(output_path, 'a').close()
    with open(output_path, 'w') as fout:
        for row in data:
            buf = ''
            for d in row:
                buf += str(d) + ' '
            fout.write(buf.rstrip() + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of initial samples on mesh edges and faces [default: 500,000]")
    parser.add_argument("--sfe", type=str, required=True, help="executable file")
    parser.add_argument("--export_curvatures", type=str2bool, default=False, help="Whether to export curvatures")
    parser.add_argument("--export_pca", type=str2bool, default=False, help="Whether to export pca")
    parser.add_argument("--override", type=str2bool, default=True, help="Whether to override tmp file")
    parser.add_argument("--remove", type=str2bool, default=True, help="Whether to remove tmp file")
    parser.add_argument("--model_id", type=str, required=True,)
    parser.add_argument("--model_filename", type=str, required=True,)
    parser.add_argument("--pts_dir", type=str, required=True,)
    parser.add_argument("--face_dir", type=str, required=True,)
    ARGS = parser.parse_args()
    print("ARGS: {}".format(ARGS))

    create_point_clouds([[ARGS.model_id, ARGS.model_filename]], 0,
                            n_samples=ARGS.num_samples, debug=True, export_curvatures=ARGS.export_curvatures,
                            export_pca=ARGS.export_pca, sfe=ARGS.sfe, override=ARGS.override, remove=ARGS.remove,
                            points_dir=ARGS.pts_dir, faces_dir=ARGS.face_dir)
