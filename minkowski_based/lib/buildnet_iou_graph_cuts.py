import os
import numpy as np
import json
import sys
from scipy import spatial
from tqdm import tqdm
from lib.mesh_utils import read_obj, read_ply, calculate_face_area
from lib.buildnet_iou import get_building_point_iou, get_building_mesh_iou, get_shape_iou, get_part_iou, get_split_models,\
  get_mesh_data, save_pred_in_json, BUILDNET_TEST_SPLIT, toplabels, classification_accuracy

# Network results directory
NET_RESULTS_DIR = sys.argv[1]
assert(os.path.isdir(NET_RESULTS_DIR))

# Create directories for best results
BEST_TRIANGLES_GC_DIR = os.path.join(NET_RESULTS_DIR, "best_triangles_gc")
os.makedirs(BEST_TRIANGLES_GC_DIR, exist_ok=True)


if __name__ == "__main__":

	top_k = 200
	best_iou_model = np.zeros((top_k,))
	best_iou_model[:] = 0.000000001
	best_model_triangles_pred, best_model_fn = [[] for _ in range(top_k)], [[] for _ in range(top_k)]

	# Get model names
	models_fn = get_split_models(split_fn=BUILDNET_TEST_SPLIT)

	mesh_buildings_iou_from_tr, mesh_buildings_acc_from_tr = {}, {}

	print("Calculate part and shape IOU for mesh tracks using graph cuts")
	for model_fn in tqdm(models_fn):
		# Get mesh data
		vertices, faces, face_gt_labels, components, face_area = get_mesh_data(model_fn)
		# Get face labels
		face_pred_labels_from_tr = np.load(os.path.join(NET_RESULTS_DIR, "tr_labels_graph_cuts", model_fn + "_labels.npy"))
		assert(face_pred_labels_from_tr.shape[0] == faces.shape[0])
		# Calculate mesh building iou
		mesh_buildings_iou_from_tr[model_fn] = get_building_mesh_iou(face_gt_labels, face_pred_labels_from_tr, face_area)
		# Calculate classification accuracy
		mesh_buildings_acc_from_tr[model_fn] = classification_accuracy(face_gt_labels, face_pred_labels_from_tr, face_area)
		# Save best and worst model
		label_iou = mesh_buildings_iou_from_tr[model_fn]["label_iou"]
		s_iou = np.sum([v for v in label_iou.values()]) / float(len(label_iou))
		if s_iou > best_iou_model[-1]:
			best_iou_model[top_k-1] = s_iou
			best_model_triangles_pred[top_k - 1] = face_pred_labels_from_tr
			best_model_fn[top_k-1] = model_fn
			sort_idx = np.argsort(1/np.asarray(best_iou_model)).tolist()
			best_iou_model = best_iou_model[sort_idx]
			best_model_triangles_pred = [best_model_triangles_pred[idx] for idx in sort_idx]
			best_model_fn = [best_model_fn[idx] for idx in sort_idx]

	# Calculate avg part and shape IOU
	mesh_shape_iou_from_tr = get_shape_iou(buildings_iou=mesh_buildings_iou_from_tr)
	mesh_part_iou_from_tr = get_part_iou(buildings_iou=mesh_buildings_iou_from_tr)
	mesh_acc_from_tr = np.sum([acc for acc in mesh_buildings_acc_from_tr.values()]) / float(len(mesh_buildings_acc_from_tr))

	# Save best
	buf = ''
	for i in range(top_k):
		buf += "Best model iou: " + str(best_iou_model[i]) + ", " + best_model_fn[i] + '\n'
		save_pred_in_json(best_model_triangles_pred[i], os.path.join(BEST_TRIANGLES_GC_DIR, best_model_fn[i] + "_label.json"))

	# Log results
	buf += "Mesh Classification Accuracy From Triangles: " + str(np.round(mesh_acc_from_tr * 100, 2)) + '\n' \
				 "Mesh Shape IoU From Triangles: " + str(np.round(mesh_shape_iou_from_tr['all'] * 100, 2)) + '\n' \
         "Mesh Part IoU From Triangles: " + str(np.round(mesh_part_iou_from_tr['all'] * 100, 2)) + '\n' \
         "Mesh Part IoU From Triangles - FR: " + str(np.round(mesh_part_iou_from_tr['fr-part'] * 100, 2)) + '\n' \
         "Per label mesh part IoU from triangles: " + ", ".join([label + ": " +
            str(np.round(mesh_part_iou_from_tr[label][0] * 100, 2)) for label in toplabels.values() if label != "undetermined"])
	print(buf)
	with open(os.path.join(NET_RESULTS_DIR, "graph_cuts_results_log.txt"), 'w') as fout_txt:
		fout_txt.write(buf)
