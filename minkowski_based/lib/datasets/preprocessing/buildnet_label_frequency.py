from pathlib import Path
import numpy as np
import json
from tqdm import tqdm

BUILDNET_BASE_PATH = Path('/media/melinos/BigData/BUILDNET/BUILDNET_2000/ShapeFeaExport_PLY')
BUILDNET_LABELS_PATH = Path('/media/melinos/BigData/BUILDNET/BUILDNET_2000/ShapeFeaExport_PLY/labels_32')
TRAIN_SPLIT_PATH = BUILDNET_BASE_PATH / 'splits' / 'train_split.txt'

MAXIMUM_LABEL_ID = 31 # Undetermined is also added
dataset_labels = np.zeros((MAXIMUM_LABEL_ID + 1,), dtype=np.int64)

# Read models in training split
model_list = []
with open(TRAIN_SPLIT_PATH, 'r') as fin:
	for line in fin:
		model_list.append(line.strip())

for model in tqdm(model_list):
	labels_path = BUILDNET_LABELS_PATH / (model + '_label.json')
	# Read labels
	with open(labels_path, 'r') as fin_json:
		labels_json = json.load(fin_json)
	labels = np.fromiter(labels_json.values(), dtype=np.int64)
	unique_labels, counts = np.unique(labels, return_counts=True)
	dataset_labels[unique_labels] += counts

assert(np.sum(dataset_labels) == labels.shape[0] * len(model_list))
# Save labels frequency
labels_json = {}
for label_ind in range(dataset_labels.shape[0]):
	labels_json[str(label_ind)] = int(dataset_labels[label_ind])
with open(BUILDNET_BASE_PATH / "training_split_labels_freq.json", 'w') as fout_json:
	json.dump(labels_json, fout_json)


