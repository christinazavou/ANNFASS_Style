import argparse
import logging
import os
import shutil
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from common.mesh_utils import ObjMeshWithComponents, centralize_and_unit_scale
from common.utils import _THR_TOL_32, _THR_TOL_64, parse_buildings_csv, str2bool, set_logger_file

LOGGER = logging.getLogger(__file__)


# TODO: move textures into one folder and make mtl files point to them
def process_building(triangulated_obj_fn, final_obj_fn, override=False):
    if not os.path.exists(triangulated_obj_fn):
        LOGGER.info("File {} doesn't exist. Won't proces.".format(triangulated_obj_fn))
        return
    if os.path.exists(final_obj_fn) and not override:
        LOGGER.info("File {} exists. Won't override.".format(final_obj_fn))
        return
    os.makedirs(os.path.dirname(final_obj_fn), exist_ok=True)
    basic_mesh = ObjMeshWithComponents(triangulated_obj_fn)
    xyz_min = basic_mesh.vertex_coords.min(axis=0)
    xyz_max = basic_mesh.vertex_coords.max(axis=0)
    centroid = (xyz_min + xyz_max) / 2.

    max_dist_to_origin = np.max(np.sqrt(np.sum(basic_mesh.vertex_coords ** 2, axis=1)))
    div_by = np.maximum(max_dist_to_origin, _THR_TOL_64 if max_dist_to_origin.dtype == np.float64 else _THR_TOL_32)

    tmp_obj_fn = triangulated_obj_fn.replace(".obj", "tmp.obj")
    centralize_and_unit_scale(triangulated_obj_fn, tmp_obj_fn, centroid, 1/div_by)

    basic_mesh = ObjMeshWithComponents(tmp_obj_fn)
    xyz_min = basic_mesh.vertex_coords.min(axis=0)
    xyz_max = basic_mesh.vertex_coords.max(axis=0)

    bbox_diag = np.array(xyz_max) - np.array(xyz_min)
    bbox_diag_len = np.sqrt(np.sum(bbox_diag**2))
    div_by = np.maximum(bbox_diag_len, _THR_TOL_64 if bbox_diag_len.dtype == np.float64 else _THR_TOL_32)

    centralize_and_unit_scale(tmp_obj_fn, final_obj_fn, np.array([0, 0, 0]), 1/div_by)
    os.remove(tmp_obj_fn)
    if os.path.exists(os.path.join(os.path.dirname(final_obj_fn), "textures")):
        shutil.rmtree(os.path.join(os.path.dirname(final_obj_fn), "textures"))
    shutil.copytree(os.path.join(os.path.dirname(triangulated_obj_fn), "textures"),
                    os.path.join(os.path.dirname(final_obj_fn), "textures"))
    shutil.copyfile(triangulated_obj_fn.replace(".obj", ".mtl"),
                    final_obj_fn.replace(".obj", ".mtl"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--obj_dir_in', type=str, default="triangledObj")
    parser.add_argument('--obj_dir_out', type=str, default="normalizedObjNew")
    parser.add_argument('--buildings_csv', type=str, required=True)
    parser.add_argument('--override', type=str2bool, default=False)
    parser.add_argument('--logs_dir', type=str, default="logs")
    args = parser.parse_args()

    logs_dir = os.path.join(args.root_dir, args.logs_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    _log_file = os.path.join(logs_dir, os.path.basename(args.buildings_csv).replace('.csv', '.log'))
    LOGGER = set_logger_file(_log_file, LOGGER)

    LOGGER.info(f"Starting running {os.path.realpath(__file__)}...")

    buildings = parse_buildings_csv(args.buildings_csv)

    for building in buildings:
        LOGGER.info("Processing building {}".format(building))
        triangulated_obj = os.path.join(args.root_dir, args.obj_dir_in, building, "{}.obj".format(building))
        final_obj = os.path.join(args.root_dir, args.obj_dir_out, building, "{}.obj".format(building))
        process_building(triangulated_obj, final_obj, args.override)

    LOGGER.info(f"End running {os.path.realpath(__file__)}")
