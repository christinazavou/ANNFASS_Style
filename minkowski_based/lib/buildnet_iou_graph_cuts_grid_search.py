import os
import numpy as np
import json
import sys
from scipy import spatial
from tqdm import tqdm
import subprocess
import shlex
from lib.buildnet_iou import get_building_mesh_iou, get_shape_iou, get_part_iou, get_split_models, get_mesh_data, \
  toplabels, BUILDNET_SPLITS_DIR

# BuildNet directories
BUILDNET_VAL_SPLIT = os.path.join(BUILDNET_SPLITS_DIR, "val_split.txt")
assert(os.path.isfile(BUILDNET_VAL_SPLIT))

# Network results directory
NET_RESULTS_DIR = sys.argv[1]
assert(os.path.isdir(NET_RESULTS_DIR))


if __name__ == "__main__":

  # Get model names
  models_fn = get_split_models(split_fn=BUILDNET_VAL_SPLIT)

  mesh_buildings_iou_from_tr, mesh_building_data, grid_search_results = {}, {}, {}

  # Clear results
  grid_search_results_fn = os.path.join(NET_RESULTS_DIR, "graph_cuts_grid_search_log.txt")
  with open(grid_search_results_fn, 'w') as fout_txt:
    fout_txt.write('')

  # Get gcuts_smoothing params from logspace
  gcuts_smoothing_params = np.logspace(-2, 1, 100)
  for gcuts_smoothing in tqdm(gcuts_smoothing_params):
    # Run graph cuts
    print("Run graph_cuts with gcuts smoothing: {}" .format(gcuts_smoothing))
    graph_cuts_CMD =  'matlab -nosplash -nodesktop -r "cd graph_cuts; run_graph_cuts_on_mesh(\'' +\
                      NET_RESULTS_DIR + '/face_feat_from_tr\',' + str(gcuts_smoothing) + '); quit"'
    proc = subprocess.Popen(shlex.split(graph_cuts_CMD), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    print("Calculate part and shape IOU for mesh tracks using graph cuts and gcuts smoothing: {}"
          .format(gcuts_smoothing))
    for model_fn in tqdm(models_fn):
      if model_fn in mesh_building_data.keys():
        vertices = mesh_building_data[model_fn]['vertices']
        faces = mesh_building_data[model_fn]['faces']
        face_gt_labels = mesh_building_data[model_fn]['face_gt_labels']
        components = mesh_building_data[model_fn]['components']
        face_area = mesh_building_data[model_fn]['face_area']
      else:
        # Get mesh data
        vertices, faces, face_gt_labels, components, face_area = get_mesh_data(model_fn)
        mesh_building_data[model_fn] = {
          'vertices': vertices,
          'faces': faces,
          'face_gt_labels': face_gt_labels,
          'components': components,
          'face_area': face_area
        }
      # Get face labels
      face_pred_labels_from_tr = np.load(os.path.join(NET_RESULTS_DIR, "tr_labels_graph_cuts", model_fn + "_labels.npy"))
      assert(face_pred_labels_from_tr.shape[0] == faces.shape[0])
      # Calculate mesh building iou
      mesh_buildings_iou_from_tr[model_fn] = get_building_mesh_iou(face_gt_labels, face_pred_labels_from_tr, face_area)

    # Calculate avg part and shape IOU
    mesh_shape_iou_from_tr = get_shape_iou(buildings_iou=mesh_buildings_iou_from_tr)
    mesh_part_iou_from_tr = get_part_iou(buildings_iou=mesh_buildings_iou_from_tr)

    # Log results
    grid_search_results[str(gcuts_smoothing)] = mesh_part_iou_from_tr['fr-part'] * 100
    with open(os.path.join(NET_RESULTS_DIR, 'grid_search_results.json'), 'w') as fout_json:
      json.dump(grid_search_results, fout_json)
    buf = "Gcuts smoothing: " + str(gcuts_smoothing) + '\n' \
          "------------------" + '\n' \
          "Mesh Shape IoU From Triangles: " + str(np.round(mesh_shape_iou_from_tr['all'] * 100, 2)) + '\n' \
          "Mesh Part IoU From Triangles: " + str(np.round(mesh_part_iou_from_tr['all'] * 100, 2)) + '\n' \
          "Mesh Part IoU From Triangles - FR: " + str(np.round(mesh_part_iou_from_tr['fr-part'] * 100, 2)) + '\n' \
          "Per label mesh part IoU from triangles: " + ", ".join([label + ": " +
             str(np.round(mesh_part_iou_from_tr[label][0] * 100, 2)) for label in toplabels.values() if label != "undetermined"]) + '\n'
    print(buf)
    with open(grid_search_results_fn, 'a') as fout_txt:
      fout_txt.write(buf)