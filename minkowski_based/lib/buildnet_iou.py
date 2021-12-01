import os
import numpy as np
import json
import sys
from scipy import spatial
from tqdm import tqdm
from lib.mesh_utils import read_obj, read_ply, calculate_face_area

# Majority labels (exclude ramp and canopy_gazebo)
toplabels = {0:"undetermined", 1:"wall", 2:"window", 3:"vehicle", 4:"roof", 5:"plant_tree", 6:"door", 7:"tower_steeple",
             8:"furniture", 9:"ground_grass", 10:"beam_frame", 11:"stairs", 12:"column", 13:"railing_baluster",
             14:"floor", 15:"chimney", 16:"ceiling", 17:"fence", 18:"pond_pool", 19:"corridor_path", 20:"balcony_patio",
             21:"garage", 22:"dome", 23:"road", 24:"entrance_gate", 25:"parapet_merlon", 26:"buttress", 27:"dormer",
             28:"lantern_lamp", 29:"arch", 30:"awning", 31:"shutters"} # , 32:"ramp", 33:"canopy_gazebo" }

# Most frequent labels
freqlabels={1:"wall", 2:"window", 3:"vehicle", 4:"roof", 5:"plant_tree", 6:"door", 7:"tower_steeple", 8:"furniture",
            10:"beam_frame", 11:"stairs", 12:"column", 17:"fence", 20:"balcony_patio", 25:"parapet_merlon"}

# BuildNet directories
BUILDNET_BASE_DIR = os.path.join(os.sep, "media", "melinos", "BigData", "BUILDNET", "BUILDNET_2000")
assert(os.path.isdir(BUILDNET_BASE_DIR))
BUILDNET_OBJ_DIR = os.path.join(BUILDNET_BASE_DIR, "flippedNormal_unit_obj_withtexture")
assert(os.path.isdir(BUILDNET_OBJ_DIR))
BUILDNET_PTS_DIR = os.path.join(BUILDNET_BASE_DIR, "ShapeFeaExport_PLY", "nocolor")
assert(BUILDNET_PTS_DIR)
BUILDNET_PTS_LABELS_DIR = os.path.join(BUILDNET_BASE_DIR, "ShapeFeaExport_PLY", "labels_32")
assert(BUILDNET_PTS_LABELS_DIR)
BUILDNET_PTS_FACEINDEX_DIR = os.path.join(BUILDNET_BASE_DIR, "ShapeFeaExport_PLY", "faceindex")
assert(os.path.isdir(BUILDNET_PTS_FACEINDEX_DIR))
BUILDNET_COMP_TO_LABELS_DIR = os.path.join(BUILDNET_BASE_DIR, "component_to_labels")
assert(os.path.isdir(BUILDNET_COMP_TO_LABELS_DIR))
BUILDNET_SPLITS_DIR = os.path.join(BUILDNET_BASE_DIR, "ShapeFeaExport_PLY", "splits")
assert(os.path.isdir(BUILDNET_SPLITS_DIR))
BUILDNET_TEST_SPLIT = os.path.join(BUILDNET_SPLITS_DIR, "test_split.txt")
assert(os.path.isfile(BUILDNET_TEST_SPLIT))

# Network results directory
NET_RESULTS_DIR = sys.argv[1]
assert(os.path.isdir(NET_RESULTS_DIR))

# Create directories for best results
BEST_POINTS_DIR = os.path.join(NET_RESULTS_DIR, "best_points")
os.makedirs(BEST_POINTS_DIR, exist_ok=True)
BEST_TRIANGLES_DIR = os.path.join(NET_RESULTS_DIR, "best_triangles")
os.makedirs(BEST_TRIANGLES_DIR, exist_ok=True)
BEST_COMP_DIR = os.path.join(NET_RESULTS_DIR, "best_comp")
os.makedirs(BEST_COMP_DIR, exist_ok=True)

# Create directories for aggregated mesh features
FACE_FEAT_FROM_TR_DIR = os.path.join(NET_RESULTS_DIR, "face_feat_from_tr")
os.makedirs(FACE_FEAT_FROM_TR_DIR, exist_ok=True)
FACE_FEAT_FROM_COMP_DIR = os.path.join(NET_RESULTS_DIR, "face_feat_from_comp")
os.makedirs(FACE_FEAT_FROM_COMP_DIR, exist_ok=True)


def classification_accuracy(ground, prediction, face_area=None):
  """
    Classification accuracy
  :param ground: N x 1, numpy.ndarray(int)
  :param prediction: N x 1, numpy.ndarray(int)
  :param face_area: N x 1, numpy.ndarray(float)
  :return:
    accuracy: float
  """

  prediction = np.copy(prediction)
  ground = np.copy(ground)
  non_zero_idx = np.squeeze(ground != 0).nonzero()[0]
  ground = ground[non_zero_idx]
  prediction = prediction[non_zero_idx]
  if face_area is not None:
    face_area = np.copy(face_area)
    face_area = face_area[non_zero_idx]
    accuracy = np.dot(face_area.T, ground == prediction)[0] / np.sum(face_area)
    accuracy = accuracy[0]
  else:
    accuracy = np.sum(ground == prediction) / float(len(ground))

  return accuracy


def get_building_point_iou(ground, prediction):
  """
    Calculate point IOU for buildings
  :param ground: N x 1, numpy.ndarray(int)
  :param prediction: N x 1, numpy.ndarray(int)
  :return:
    metrics: dict: {
                    "label_iou": dict{label: iou (float)},
                    "intersection": dict{label: intersection (float)},
                    "union": dict{label: union (float)
                   }
  """

  label_iou, intersection, union = {}, {}, {}
  # Ignore undetermined
  prediction = np.copy(prediction)
  prediction[ground == 0] = 0

  for i in range(1, len(toplabels)):
    # Calculate intersection and union for ground truth and predicted labels
    intersection_i = np.sum((ground == i) & (prediction== i))
    union_i = np.sum((ground == i) | (prediction == i))

    # If label i is present either on the gt or the pred set
    if union_i > 0:
      intersection[i] = float(intersection_i)
      union[i] = float(union_i)
      label_iou[i] = intersection[i] / union[i]

  metrics = {"label_iou": label_iou, "intersection": intersection, "union": union}

  return metrics


def get_building_mesh_iou(ground, prediction, face_area):
  """
    Calculate mesh IOU for buildings
  :param ground: N x 1, numpy.ndarray(int)
  :param prediction: N x 1, numpy.ndarray(int)
  :param face_area: N x 1, numpy.ndarray(float)
  :return:
    metrics: dict: {
                    "label_iou": dict{label: iou (float)},
                    "intersection": dict{label: intersection (float)},
                    "union": dict{label: union (float)
                   }
  """

  label_iou, intersection, union = {}, {}, {}
  # Ignore undetermined
  prediction = np.copy(prediction)
  prediction[ground == 0] = 0

  for i in range(1, len(toplabels)):
    # Calculate binary intersection and union for ground truth and predicted labels
    intersection_i = ((ground == i) & (prediction == i))
    union_i = ((ground == i) | (prediction == i))

    if np.sum(union_i) > 0:
      intersection[i] = np.dot(face_area.T, intersection_i)[0]
      union[i] = np.dot(face_area.T, union_i)[0]
      if union[i] > 0.0:
        label_iou[i] = intersection[i] / union[i]
      else:
        print(len(union[i]))
        label_iou[i] = np.array([0.0])

  metrics = {"label_iou": label_iou, "intersection": intersection, "union": union}

  return metrics


def get_shape_iou(buildings_iou):
  """
    Average label IOU and calculate overall shape IOU
  :param buildings_iou: dict: {
                                <model_name> : {
                                                "label_iou": dict{label: iou (float)},
                                                "intersection": dict{label: intersection (float)},
                                                "union": dict{label: union (float)
                                               }
                              }
  :return:
    shape_iou: dict: {
                      "all": avg shape iou,
                      <model_name>: per building shape iou
                     }
  """

  shape_iou = {}

  for building, metrics in buildings_iou.items():
    # Average label iou per shape
    L_s = len(metrics["label_iou"])
    shape_iou[building] = np.sum([v for v in metrics["label_iou"].values()]) / float(L_s)

  # Dataset avg shape iou
  shape_iou['all'] = np.sum([v for v in shape_iou.values()]) / float(len(buildings_iou))

  return shape_iou


def get_part_iou(buildings_iou):
  """
    Average intersection/union and calculate overall part IOU and most frequent part IOU
  :param buildings_iou: dict: {
                              <model_name> : {
                                              "label_iou": dict{label: iou (float)},
                                              "intersection": dict{label: intersection (float)},
                                              "union": dict{label: union (float)
                                             }
                             }
  :return:
    part_iou:  dict: {
                      "all": avg part iou,
                      "fr-part": most frequent labels part iou
                      <label_name>: per label part iou
                     }
  """

  intersection = {i: 0.0 for i in range(1, len(toplabels))}
  union = {i: 0.0 for i in range(1, len(toplabels))}

  for building, metrics in buildings_iou.items():
    for label in metrics["intersection"].keys():
      # Accumulate intersection and union for each label across all shapes
      intersection[label] += metrics["intersection"][label]
      union[label] += metrics["union"][label]

  # Calculate part IOU for each label
  part_iou = {toplabels[key]: intersection[key] / union[key] for key in range(1, len(toplabels))}
  # Avg part IOU
  part_iou["all"] = np.sum([v for v in part_iou.values()]) / float(len(toplabels) - 1)
  # Most frequent labels part IOU
  part_iou["fr-part"] = np.sum([part_iou[freqlabels[key]] for key in freqlabels.keys()]) / float(len(freqlabels))

  return part_iou


def transfer_point_predictions(vertices, faces, components, points, point_feat, point_face_index, max_pool=False):
  """
    Transfer point predictions to triangles and components through avg pooling
  :param vertices: N x 3, numpy.ndarray(float)
  :param faces: M x 3, numpy.ndarray(int)
  :param components: M x 1, numpy.ndarray(int)
  :param points: K x 3, numpy.ndarray(float)
  :param point_feat: K x 31, numpy.ndarray(float)
  :param point_face_index: K x 1, numpy.ndarray(int)
  :param max_pool: bool
  :return:
    face_labels_from_triangle_avg_pool: M x 1, numpy.ndarray(int)
    face_labels_from_comp_avg_pool: M x 1, numpy.ndarray(int)
    face_feat_from_tr_avg_pool: M x 31, numpy.ndarray(float)
    face_feat_from_comp_avg_pool: M x 31, numpy.ndarray(float)
    face_labels_from_triangle_max_pool: M x 1, numpy.ndarray(int)
    face_labels_from_comp_max_pool: M x 1, numpy.ndarray(int)
  """

  n_components = len(np.unique(components))
  face_feat_from_tr_avg_pool = np.zeros((faces.shape[0], point_feat.shape[1]))
  face_feat_from_comp_avg_pool = np.zeros((faces.shape[0], point_feat.shape[1]))
  comp_feat_avg_pool = np.zeros((n_components, point_feat.shape[1]))
  if max_pool:
    face_feat_from_tr_max_pool = np.zeros_like(face_feat_from_tr_avg_pool)
    face_feat_from_comp_max_pool = np.zeros_like(face_feat_from_comp_avg_pool)
    comp_feat_max_pool = np.zeros_like(comp_feat_avg_pool)
  face_point_index = {}

  # Find faces that have no corresponding points
  sampled = set(point_face_index.flatten())
  unsampled = list(set(np.arange(len(faces))) - sampled)  # faces with no sample points

  # Compute center for unsampled faces
  v0 = vertices[faces[unsampled, 0]]
  v1 = vertices[faces[unsampled, 1]]
  v2 = vertices[faces[unsampled, 2]]
  face_centers = np.array([[v0[:, 0] + v1[:, 0] + v2[:, 0]],
                           [v0[:, 1] + v1[:, 1] + v2[:, 1]],
                           [v0[:, 2] + v1[:, 2] + v2[:, 2]]]) / 3.0
  face_centers = np.squeeze(face_centers).T

  ## Transfer point predictions to triangles
  # Find nearest point and assign its point feature to each unsampled face
  p_tree = spatial.cKDTree(points)
  _, k_nn_idx = p_tree.query(face_centers)
  for idx, face in enumerate(unsampled):
    face_feat_from_tr_avg_pool[face] = point_feat[k_nn_idx[idx]]
    if max_pool:
      face_feat_from_tr_max_pool[face] = point_feat[k_nn_idx[idx]]
    face_point_index[face] = int(k_nn_idx[idx])

  # Use avg pooling for sampled faces
  for face in sampled:
    mask = np.squeeze(point_face_index == face)
    face_feat_from_tr_avg_pool[face] = np.mean(point_feat[mask], axis=0)
    if max_pool:
      # Use max pooling also
      face_feat_from_tr_max_pool[face] = np.amax(point_feat[mask], axis=0)
    face_point_index[face] = mask.nonzero()[0].tolist()

  ## Transfer point predictions to components
  for comp_idx in range(comp_feat_avg_pool.shape[0]):
    face_idx = np.squeeze(components == comp_idx).nonzero()[0]
    point_idx = []
    for idx in face_idx:
      try:
        point_idx.extend(face_point_index[int(idx)])
      except:
        point_idx.append(face_point_index[int(idx)])
    comp_feat_avg_pool[comp_idx] = np.mean(point_feat[point_idx], axis=0)
    face_feat_from_comp_avg_pool[face_idx] = comp_feat_avg_pool[comp_idx]
    if max_pool:
      comp_feat_max_pool[comp_idx] = np.amax(point_feat[point_idx], axis=0)
      face_feat_from_comp_max_pool[face_idx] = comp_feat_max_pool[comp_idx]

  face_labels_from_tr_avg_pool = np.argmax(face_feat_from_tr_avg_pool, axis=1)[:, np.newaxis] + 1  # we exclude undetermined (label 0) during training
  face_labels_from_comp_avg_pool = np.argmax(face_feat_from_comp_avg_pool, axis=1)[:, np.newaxis] + 1

  if max_pool:
    face_labels_from_tr_max_pool = np.argmax(face_feat_from_tr_max_pool, axis=1)[:, np.newaxis] + 1
    face_labels_from_comp_max_pool = np.argmax(face_feat_from_comp_max_pool, axis=1)[:, np.newaxis] + 1
    return face_labels_from_tr_avg_pool, face_labels_from_comp_avg_pool, face_feat_from_tr_avg_pool, \
           face_feat_from_comp_avg_pool, face_labels_from_tr_max_pool, face_labels_from_comp_max_pool

  return face_labels_from_tr_avg_pool, face_labels_from_comp_avg_pool, face_feat_from_tr_avg_pool, \
         face_feat_from_comp_avg_pool


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


def get_point_cloud_data(model_name):
  """
    Get point cloud data needed for evaluation
  :param model_name: str
  :return:
    points: N x 3, numpy.ndarray(float)
    point_gt_labels: N x 1, numpy.ndarray(int)
    point_pred_labels: N x 1, numpy.ndarray(int)
    point_pred_feat: N x 31, numpy.ndarray(float)
    point_face_index: N x 1, numpy.ndarray(int)
  """

  # Get points
  points, _ = read_ply(os.path.join(BUILDNET_PTS_DIR, model_name + ".ply"))

  # Get ground truth labels
  with open(os.path.join(BUILDNET_PTS_LABELS_DIR, model_name + "_label.json"), 'r') as fin_json:
    labels_json = json.load(fin_json)
  point_gt_labels = np.fromiter(labels_json.values(), dtype=int)[:, np.newaxis]
  assert(points.shape[0] == point_gt_labels.shape[0])

  # Get per point features
  point_feat = np.load(os.path.join(NET_RESULTS_DIR, model_fn + ".npy"))
  assert(point_feat.shape[0] == point_gt_labels.shape[0])
  assert(point_feat.shape[1] == (len(toplabels)-1))

  # Calculate pred label
  point_pred_labels = np.argmax(point_feat, axis=1)[:, np.newaxis] + 1  # we exclude undetermined (label 0) during training
  assert(point_gt_labels.shape == point_pred_labels.shape)

  # Get points face index
  with open(os.path.join(BUILDNET_PTS_FACEINDEX_DIR, model_name + ".txt"), 'r') as fin_txt:
    point_face_index = fin_txt.readlines()
  point_face_index = np.asarray([int(p.strip()) for p in point_face_index], dtype=int)[:, np.newaxis]
  assert(point_face_index.shape == point_gt_labels.shape)

  return points, point_gt_labels, point_pred_labels, point_feat, point_face_index


def get_mesh_data(model_name):
  """
    Get mesh data needed for evaluation
  :param model_name: str
  :return:
    vertices: N x 3, numpy.ndarray(float)
    faces: M x 3, numpy.ndarray(int)
    face_labels: M x 1, numpy.ndarray(int)
    components: M x 1, numpy.ndarray(float)
    face_area: M x 1, numpy.ndarray(float)
  """

  # Load obj
  vertices, faces, components = read_obj(obj_fn=os.path.join(BUILDNET_OBJ_DIR, model_name + ".obj"))

  # Calculate face area
  faces -= 1
  face_area = calculate_face_area(vertices=vertices, faces=faces)
  assert(face_area.shape[0] == faces.shape[0])

  # Read components to labels
  with open(os.path.join(BUILDNET_COMP_TO_LABELS_DIR, model_name + "_label.json"), 'r') as fin_json:
    labels_json = json.load(fin_json)
  face_labels = np.zeros_like(components)
  for comp, label in labels_json.items():
    face_labels[np.where(components == int(comp))[0]] = label

  return vertices, faces, face_labels, components, face_area


def save_pred_in_json(labels, fn_json):
  """
    Save labels in json format
  :param labels: N x 1, numpy.ndarray(int)
  :param fn_json: str
  :return:
    None
  """

  # Convert numpy to dict
  labels_json = dict(zip(np.arange(labels.shape[0]).astype(str), np.squeeze(labels).tolist()))
  # Export json file
  with open(fn_json, 'w') as fout_json:
    json.dump(labels_json, fout_json)


if __name__ == "__main__":

  top_k = 5
  best_iou_model = np.zeros((top_k,))
  best_iou_model[:] = 0.000000001
  best_model_points_pred, best_model_triangles_pred, best_model_comp_pred, best_model_fn = [[] for _ in range(top_k)], \
                                                                                           [[] for _ in range(top_k)], \
                                                                                           [[] for _ in range(top_k)], \
                                                                                           [[] for _ in range(top_k)]

  # Get model names
  models_fn = get_split_models(split_fn=BUILDNET_TEST_SPLIT)

  point_buildings_iou, mesh_buildings_iou_from_tr, mesh_buildings_iou_from_comp, mesh_buildings_iou_from_tr_max_pool, \
  mesh_buildings_iou_from_comp_max_pool = {}, {}, {}, {}, {}
  point_buildings_acc, mesh_buildings_acc_from_tr, mesh_buildings_acc_from_comp, mesh_buildings_acc_from_tr_max_pool, \
  mesh_buildings_acc_from_comp_max_pool = {}, {}, {}, {}, {}

  print("Calculate part and shape IOU for point and mesh tracks")
  for model_fn in tqdm(models_fn):
    # Get point cloud data
    points, point_gt_labels, point_pred_labels, point_feat, point_face_index = get_point_cloud_data(model_fn)
    # Get mesh data
    vertices, faces, face_gt_labels, components, face_area = get_mesh_data(model_fn)
    # Infer face labels from point predictions
    face_pred_labels_from_tr, face_pred_labels_from_comp, face_feat_from_tr, face_feat_from_comp, \
    face_pred_labels_from_tr_max_pool, face_pred_labels_from_comp_max_pool = \
      transfer_point_predictions(vertices, faces, components, points, point_feat, point_face_index, max_pool=True)
    # Calculate point building iou
    point_buildings_iou[model_fn] = get_building_point_iou(point_gt_labels, point_pred_labels)
    # Calculate mesh building iou
    mesh_buildings_iou_from_tr[model_fn] = get_building_mesh_iou(face_gt_labels, face_pred_labels_from_tr, face_area)
    mesh_buildings_iou_from_comp[model_fn] = get_building_mesh_iou(face_gt_labels, face_pred_labels_from_comp, face_area)
    mesh_buildings_iou_from_tr_max_pool[model_fn] = \
      get_building_mesh_iou(face_gt_labels, face_pred_labels_from_tr_max_pool, face_area)
    mesh_buildings_iou_from_comp_max_pool[model_fn] = \
      get_building_mesh_iou(face_gt_labels, face_pred_labels_from_comp_max_pool, face_area)
    # Calculate classification accuracy
    point_buildings_acc[model_fn] = classification_accuracy(point_gt_labels, point_pred_labels)
    mesh_buildings_acc_from_tr[model_fn] = classification_accuracy(face_gt_labels, face_pred_labels_from_tr, face_area)
    mesh_buildings_acc_from_comp[model_fn] = classification_accuracy(face_gt_labels, face_pred_labels_from_comp, face_area)
    mesh_buildings_acc_from_tr_max_pool[model_fn] = \
      classification_accuracy(face_gt_labels, face_pred_labels_from_tr_max_pool, face_area)
    mesh_buildings_acc_from_comp_max_pool[model_fn] = \
      classification_accuracy(face_gt_labels, face_pred_labels_from_comp_max_pool, face_area)
    # Save mesh feat data
    np.save(os.path.join(FACE_FEAT_FROM_TR_DIR, model_fn + ".npy"), face_feat_from_tr.astype(np.float32))
    np.save(os.path.join(FACE_FEAT_FROM_COMP_DIR, model_fn + ".npy"), face_feat_from_comp.astype(np.float32))

    # Save best and worst model
    label_iou = mesh_buildings_iou_from_comp[model_fn]["label_iou"]
    s_iou = np.sum([v for v in label_iou.values()]) / float(len(label_iou))
    if s_iou > best_iou_model[-1]:
      best_iou_model[top_k-1] = s_iou
      best_model_points_pred[top_k - 1] = point_pred_labels
      best_model_triangles_pred[top_k - 1] = face_pred_labels_from_tr
      best_model_comp_pred[top_k-1] = face_pred_labels_from_comp
      best_model_fn[top_k-1] = model_fn
      sort_idx = np.argsort(1/np.asarray(best_iou_model)).tolist()
      best_iou_model = best_iou_model[sort_idx]
      best_model_points_pred = [best_model_points_pred[idx] for idx in sort_idx]
      best_model_triangles_pred = [best_model_triangles_pred[idx] for idx in sort_idx]
      best_model_comp_pred = [best_model_comp_pred[idx] for idx in sort_idx]
      best_model_fn = [best_model_fn[idx] for idx in sort_idx]

  # Calculate avg point part and shape IOU
  point_shape_iou = get_shape_iou(buildings_iou=point_buildings_iou)
  point_part_iou = get_part_iou(buildings_iou=point_buildings_iou)
  mesh_shape_iou_from_tr = get_shape_iou(buildings_iou=mesh_buildings_iou_from_tr)
  mesh_part_iou_from_tr = get_part_iou(buildings_iou=mesh_buildings_iou_from_tr)
  mesh_shape_iou_from_comp = get_shape_iou(buildings_iou=mesh_buildings_iou_from_comp)
  mesh_part_iou_from_comp = get_part_iou(buildings_iou=mesh_buildings_iou_from_comp)
  mesh_shape_iou_from_tr_max_pool = get_shape_iou(buildings_iou=mesh_buildings_iou_from_tr_max_pool)
  mesh_part_iou_from_tr_max_pool = get_part_iou(buildings_iou=mesh_buildings_iou_from_tr_max_pool)
  mesh_shape_iou_from_comp_max_pool = get_shape_iou(buildings_iou=mesh_buildings_iou_from_comp_max_pool)
  mesh_part_iou_from_comp_max_pool = get_part_iou(buildings_iou=mesh_buildings_iou_from_comp_max_pool)
  point_acc = np.sum([acc for acc in point_buildings_acc.values()]) / float(len(point_buildings_acc))
  mesh_acc_from_tr = np.sum([acc for acc in mesh_buildings_acc_from_tr.values()]) / float(len(mesh_buildings_acc_from_tr))
  mesh_acc_from_comp = np.sum([acc for acc in mesh_buildings_acc_from_comp.values()]) / float(len(mesh_buildings_acc_from_comp))
  mesh_acc_from_tr_max_pool = np.sum([acc for acc in mesh_buildings_acc_from_tr_max_pool.values()]) / float(len(mesh_buildings_acc_from_tr_max_pool))
  mesh_acc_from_comp_max_pool = np.sum([acc for acc in mesh_buildings_acc_from_comp_max_pool.values()]) / float(len(mesh_buildings_acc_from_comp_max_pool))

  # Save best
  buf = ''
  for i in range(top_k):
    buf += "Best model iou: " + str(best_iou_model[i]) + ", " + best_model_fn[i] + '\n'
    save_pred_in_json(best_model_points_pred[i], os.path.join(BEST_POINTS_DIR, best_model_fn[i] + "_label.json"))
    save_pred_in_json(best_model_triangles_pred[i], os.path.join(BEST_TRIANGLES_DIR, best_model_fn[i] + "_label.json"))
    save_pred_in_json(best_model_comp_pred[i], os.path.join(BEST_COMP_DIR, best_model_fn[i] + "_label.json"))


  # Log results
  buf += "Point Classification Accuracy: " + str(np.round(point_acc * 100, 2)) + '\n' \
         "Point Shape IoU: " + str(np.round(point_shape_iou['all'] * 100, 2)) + '\n' \
         "Point Part IoU: " + str(np.round(point_part_iou['all'] * 100, 2)) + '\n' \
         "Point Part IoU - FR: " + str(np.round(point_part_iou['fr-part'] * 100, 2)) + '\n' \
          "Per label point part IoU: " + ", ".join([label + ": " +
            str(np.round(point_part_iou[label] * 100, 2)) for label in toplabels.values() if label != "undetermined"]) + '\n' \
          "Average Pooling" + '\n' \
         "---------------" + '\n' \
         "Mesh Classification Accuracy From Triangles: " + str(np.round(mesh_acc_from_tr * 100, 2)) + '\n' \
         "Mesh Shape IoU From Triangles: " + str(np.round(mesh_shape_iou_from_tr['all'] * 100, 2)) + '\n' \
         "Mesh Part IoU From Triangles: " + str(np.round(mesh_part_iou_from_tr['all'] * 100, 2)) + '\n' \
         "Mesh Part IoU From Triangles - FR: " + str(np.round(mesh_part_iou_from_tr['fr-part'] * 100, 2)) + '\n' \
         "Mesh Classification Accuracy From Comp: " + str(np.round(mesh_acc_from_comp * 100, 2)) + '\n' \
         "Mesh Shape IoU From Comp: " + str(np.round(mesh_shape_iou_from_comp['all'] * 100, 2)) + '\n' \
         "Mesh Part IoU From Comp: " + str(np.round(mesh_part_iou_from_comp['all'] * 100, 2)) + '\n' \
         "Mesh Part IoU From Comp- FR: " + str(np.round(mesh_part_iou_from_comp['fr-part'] * 100, 2)) + '\n' \
         "Per label mesh part IoU from triangles: " + ", ".join([label + ": " +
            str(np.round(mesh_part_iou_from_tr[label][0] * 100, 2)) for label in toplabels.values() if label != "undetermined"]) + '\n' \
         "Per label mesh part IoU from comp: " + ", ".join([label + ": " +
            str(np.round(mesh_part_iou_from_comp[label][0] * 100, 2)) for label in toplabels.values() if label != "undetermined"]) + '\n' \
         "Max Pooling" + '\n' \
         "-----------" + '\n' \
         "Mesh Classification Accuracy From Triangles: " + str(np.round(mesh_acc_from_tr_max_pool * 100, 2)) + '\n' \
         "Mesh Shape IoU From Triangles: " + str(np.round(mesh_shape_iou_from_tr_max_pool['all'] * 100, 2)) + '\n' \
         "Mesh Part IoU From Triangles: " + str(np.round(mesh_part_iou_from_tr_max_pool['all'] * 100, 2)) + '\n' \
         "Mesh Part IoU From Triangles - FR: " + str( np.round(mesh_part_iou_from_tr_max_pool['fr-part'] * 100, 2)) + '\n' \
         "Mesh Classification Accuracy From Comp: " + str(np.round(mesh_acc_from_comp_max_pool * 100, 2)) + '\n' \
         "Mesh Shape IoU From Comp: " + str(np.round(mesh_shape_iou_from_comp_max_pool['all'] * 100, 2)) + '\n' \
         "Mesh Part IoU From Comp: " + str(np.round(mesh_part_iou_from_comp_max_pool['all'] * 100, 2)) + '\n' \
         "Mesh Part IoU From Comp- FR: " + str(np.round(mesh_part_iou_from_comp_max_pool['fr-part'] * 100, 2)) + '\n' \
         "Per label mesh part IoU from triangles: " + ", ".join([label + ": " +
            str(np.round(mesh_part_iou_from_tr_max_pool[label][0] * 100, 2)) for label in toplabels.values() if label != "undetermined"]) + '\n' \
         "Per label mesh part IoU from comp: " + ", ".join([label + ": " +
            str(np.round(mesh_part_iou_from_comp_max_pool[label][0] * 100, 2)) for label in toplabels.values() if label != "undetermined"]) + '\n'

  print(buf)
  with open(os.path.join(NET_RESULTS_DIR, "results_log.txt"), 'w') as fout_txt:
    fout_txt.write(buf)
