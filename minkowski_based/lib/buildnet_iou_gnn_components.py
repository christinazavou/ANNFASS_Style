import os
import numpy as np
import json
import sys
from scipy import spatial
from tqdm import tqdm
from lib.mesh_utils import read_obj, read_ply, calculate_face_area
from lib.buildnet_iou import get_building_mesh_iou, get_shape_iou, get_part_iou, get_split_models, get_mesh_data, \
  save_pred_in_json, BUILDNET_TEST_SPLIT, toplabels

# Network results directory
NET_RESULTS_DIR = sys.argv[1]
assert(os.path.isdir(NET_RESULTS_DIR))

# Create directories for best results
BEST_COMP_DIR = os.path.join(NET_RESULTS_DIR, "best_comp")
os.makedirs(BEST_COMP_DIR, exist_ok=True)


if __name__ == "__main__":

  top_k = 5
  best_iou_model = np.zeros((top_k,))
  best_iou_model[:] = 0.000000001
  best_model_comp_pred, best_model_fn = [[] for _ in range(top_k)], [[] for _ in range(top_k)]

  # Get model names
  models_fn = get_split_models(split_fn=BUILDNET_TEST_SPLIT)

  mesh_buildings_iou_from_comp = {}

  print("Calculate part and shape IOU for GNN components")
  for model_fn in tqdm(models_fn):
    # Get mesh data
    vertices, faces, face_gt_labels, components, face_area = get_mesh_data(model_fn)
    # Get components labels
    comp_labels = np.load(os.path.join(NET_RESULTS_DIR, model_fn + ".npy")) + 1
    assert(comp_labels.shape[0] == np.unique(components).shape[0])
    face_pred_labels_from_comp = np.zeros_like(face_gt_labels)
    for comp_idx, comp_label in enumerate(comp_labels):
      face_idx = np.squeeze(components == comp_idx).nonzero()[0]
      face_pred_labels_from_comp[face_idx] = comp_label
    # Calculate mesh building iou
    mesh_buildings_iou_from_comp[model_fn] = get_building_mesh_iou(face_gt_labels, face_pred_labels_from_comp, face_area)

    # Save best and worst model
    label_iou = mesh_buildings_iou_from_comp[model_fn]["label_iou"]
    s_iou = np.sum([v for v in label_iou.values()]) / float(len(label_iou))
    if s_iou > best_iou_model[-1]:
      best_iou_model[top_k-1] = s_iou
      best_model_comp_pred[top_k-1] = face_pred_labels_from_comp
      best_model_fn[top_k-1] = model_fn
      sort_idx = np.argsort(1/np.asarray(best_iou_model)).tolist()
      best_iou_model = best_iou_model[sort_idx]
      best_model_comp_pred = [best_model_comp_pred[idx] for idx in sort_idx]
      best_model_fn = [best_model_fn[idx] for idx in sort_idx]

  # Calculate avg part and shape IOU
  mesh_shape_iou_from_comp = get_shape_iou(buildings_iou=mesh_buildings_iou_from_comp)
  mesh_part_iou_from_comp = get_part_iou(buildings_iou=mesh_buildings_iou_from_comp)

  # Save best
  buf = ''
  for i in range(top_k):
    buf += "Best model iou: " + str(best_iou_model[i]) + ", " + best_model_fn[i] + '\n'
    save_pred_in_json(best_model_comp_pred[i], os.path.join(BEST_COMP_DIR, best_model_fn[i] + "_label.json"))


  # Log results
  buf += "Mesh Shape IoU From Comp: " + str(np.round(mesh_shape_iou_from_comp['all'] * 100, 2)) + '\n' \
         "Mesh Part IoU From Comp: " + str(np.round(mesh_part_iou_from_comp['all'] * 100, 2)) + '\n' \
         "Mesh Part IoU From Comp- FR: " + str(np.round(mesh_part_iou_from_comp['fr-part'] * 100, 2)) + '\n' \
         "Per label mesh part IoU from comp: " + ", ".join([label + ": " +
            str(np.round(mesh_part_iou_from_comp[label][0] * 100, 2)) for label in toplabels.values() if label != "undetermined"]) + '\n'
  print(buf)
  with open(os.path.join(NET_RESULTS_DIR, "results_log.txt"), 'w') as fout_txt:
    fout_txt.write(buf)