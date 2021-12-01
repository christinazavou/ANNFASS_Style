import os
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from lib.pc_utils import read_plyfile
import lib.transforms as t
from lib.utils import calculate_iou, calculate_shape_iou, calculate_part_iou
np.seterr('raise')

NUM_SEG = {'Bed': 15,
           'Bottle': 9,
           'Chair': 39,
           'Clock': 11,
           'Dishwasher': 7,
           'Display': 4,
           'Door': 5,
           'Earphone': 10,
           'Faucet': 12,
           'Knife': 10,
           'Lamp': 41,
           'Microwave': 6,
           'Refrigerator': 7,
           'StorageFurniture': 24,
           'Table': 51,
           'TrashCan': 11,
           'Vase': 6}


# PartNet directories
CATEGORY = sys.argv[1]
PARTNET_BASE_DIR = os.path.join(os.sep, "media", "melinos", "BigData", "ShapeNetRAW", "partnet_dataset")
assert(os.path.isdir(PARTNET_BASE_DIR))
PARTNET_PLY_DIR = os.path.join(PARTNET_BASE_DIR, "minkowski_net", CATEGORY)
assert(os.path.isdir(PARTNET_PLY_DIR))
PARTNET_TEST_SPLIT = os.path.join(PARTNET_PLY_DIR, "test.txt")
assert(os.path.isfile(PARTNET_TEST_SPLIT))

# Network results directory
NET_RESULTS_DIR = sys.argv[2]
assert(os.path.isdir(NET_RESULTS_DIR))

# Create directories for best results
BEST_POINTS_DIR = os.path.join(NET_RESULTS_DIR, "best_points")
os.makedirs(BEST_POINTS_DIR, exist_ok=True)

# Init labels and color map
NUM_LABELS = NUM_SEG[CATEGORY]
cmap = plt.cm.get_cmap("hsv", NUM_LABELS)
PARTNET_COLOR_MAP = dict(zip(range(NUM_LABELS),[tuple([int(cmap(i)[0]*255), int(cmap(i)[1]*255), int(cmap(i)[2]*255)])
                                                for i in range(NUM_LABELS)]))

def get_split_models(split_fn):
  """
    Read split.txt file and return model names
  :param split_fn:
  :return:
    models_fn: list(str)
  """

  models_fn = []
  with open(split_fn, 'r') as fin:
    for line in fin:
      models_fn.append(line.strip())

  return models_fn


if __name__ == "__main__":

  top_k = 5
  best_iou_model = np.zeros((top_k,))
  best_iou_model[:] = 0.000000001
  best_model_points_pred, best_model_fn = [[] for _ in range(top_k)], [[] for _ in range(top_k)]
  ious = {}
  # Get model names
  models_fn = get_split_models(split_fn=PARTNET_TEST_SPLIT)

  print("Calculate part and shape IoU")
  for model_fn in tqdm(models_fn):
    # Get point cloud data
    filename = os.path.join(PARTNET_PLY_DIR, model_fn)
    pointcloud = read_plyfile(filename)
    points = pointcloud[:, :3]
    points = t.normalize_coords(points)
    point_gt_labels = pointcloud[:, -1][:, np.newaxis]
    # Get per point features
    point_feat = np.load(os.path.join(NET_RESULTS_DIR, os.path.basename(model_fn)[:-4] + ".npy"))
    assert(point_feat.shape[0] == point_gt_labels.shape[0])
    assert(point_feat.shape[1] == NUM_LABELS)
    # Calculate pred label
    point_pred_labels = np.argmax(point_feat, axis=1)[:, np.newaxis]
    assert (point_gt_labels.shape == point_pred_labels.shape)
    # Calculate iou
    if not np.array_equal(np.unique(point_gt_labels), np.array([0])):
      ious[model_fn] = calculate_iou(ground=point_gt_labels, prediction=point_pred_labels, num_labels=NUM_LABELS)

      # Save best and worst model
      label_iou = ious[model_fn]["label_iou"]
      s_iou = np.nan_to_num(np.sum([v for v in label_iou.values()]) / float(len(label_iou)))
      if s_iou > best_iou_model[-1]:
        best_iou_model[top_k-1] = s_iou
        best_model_points_pred[top_k - 1] = point_pred_labels
        best_model_fn[top_k-1] = model_fn
        sort_idx = np.argsort(1/np.asarray(best_iou_model)).tolist()
        best_iou_model = best_iou_model[sort_idx]
        best_model_points_pred = [best_model_points_pred[idx] for idx in sort_idx]
        best_model_fn = [best_model_fn[idx] for idx in sort_idx]

  # Calculate avg point part and shape IoU
  shape_iou = calculate_shape_iou(ious=ious) * 100
  part_iou = calculate_part_iou(ious=ious, num_labels=NUM_LABELS) * 100

  # # Save best
  # buf = ''
  # for i in range(top_k):
  #   buf += "Best model iou: " + str(best_iou_model[i]) + ", " + best_model_fn[i] + '\n'
  #   save_pred_in_json(best_model_points_pred[i], os.path.join(BEST_POINTS_DIR, best_model_fn[i] + "_label.json"))
  #   save_pred_in_json(best_model_triangles_pred[i], os.path.join(BEST_TRIANGLES_DIR, best_model_fn[i] + "_label.json"))
  #   save_pred_in_json(best_model_comp_pred[i], os.path.join(BEST_COMP_DIR, best_model_fn[i] + "_label.json"))


  # Log results
  buf = "Shape IoU: " + str(np.round(shape_iou, 2)) + '\n' \
        "Part IoU: " + str(np.round(part_iou, 2))

  print(buf)
  with open(os.path.join(NET_RESULTS_DIR, "results_log.txt"), 'w') as fout_txt:
    fout_txt.write(buf)
