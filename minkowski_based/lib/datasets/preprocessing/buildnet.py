from pathlib import Path
import numpy as np
from lib.pc_utils import read_plyfile, save_point_cloud, save_point_cloud_with_normals
import json

BUILDNET_BASE_PATH = Path('/media/melinos/BigData/BUILDNET/BUILDNET_2000/ShapeFeaExport_PLY')
BUILDNET_OUT_PATH = Path('/media/melinos/BigData/BUILDNET/BUILDNET_2000/ShapeFeaExport_PLY/minkowski_net')

SPLITS = {"train": "withcolor",
		  "test": "withcolor",
		  "val": "withcolor"}

POINTCLOUD_FILE = '.ply'

fix = True

MAXIMUM_LABEL_ID = 31

split_path = BUILDNET_BASE_PATH / 'splits'
for out_path, in_path in SPLITS.items():
	# Read shapes in split
	split_file = split_path / (out_path + '_split.txt')
	shape_list = []
	with open(split_file, 'r') as fin:
		for line in fin:
			shape_list.append(line.strip())

	split_out_path = BUILDNET_OUT_PATH / out_path
	split_out_path.mkdir(parents=True, exist_ok=True)

	# Rewrite split list
	with open(BUILDNET_OUT_PATH / (out_path + '.txt'), 'w') as fout_split:
		for shape in shape_list:
			fout_split.write(out_path + '/' + shape + '.ply\n')

	# Read shape from split
	for ind, shape in enumerate(shape_list):
		print("Preprocess {shape:s} from split {split:s} ({ind:d}/{total:d})"
			  .format(shape=shape, split=out_path, ind=ind+1, total=len(shape_list)))
		withcolor_path = BUILDNET_BASE_PATH / 'withcolor' / (shape + '.ply')
		nocolor_path = BUILDNET_BASE_PATH / 'nocolor' / (shape + '.ply')
		if fix:
			# Fix number of points in withcolor.ply
			with open(withcolor_path, 'r') as fin:
				raw_data = fin.readlines()
			assert len(raw_data) == 100016
			rewrite = False
			if raw_data[2] != "element vertex 100000\n":
				raw_data[2] = "element vertex 100000\n"
				rewrite = True
			if raw_data[9] == "property uchar red\n":
				raw_data[9] = "property float red\n"
				rewrite = True
			if raw_data[10] == "property uchar green\n":
				raw_data[10] = "property float green\n"
				rewrite = True
			if raw_data[11] == "property uchar blue\n":
				raw_data[11] = "property float blue\n"
				rewrite = True
			if raw_data[12] == "property uchar alpha\n":
				raw_data[12] = "property float alpha\n"
				rewrite = True
			if rewrite:
				with open(withcolor_path, 'w') as fout:
					fout.writelines(raw_data)

		# Sanity check with nocolor
		shape_withcolor = read_plyfile(withcolor_path)
		shape_nocolor = read_plyfile(nocolor_path)
		assert shape_withcolor.shape[0] == shape_nocolor.shape[0]
		# Check points if are the same
		assert np.allclose(shape_withcolor[:, :3], shape_nocolor[:, :3])
		# Check normals if are the same
		assert np.allclose(shape_withcolor[:, 3:6], shape_nocolor[:, 3:6])

		# Read labels
		labels_path = BUILDNET_BASE_PATH / 'labels_32' / (shape + '_label.json')
		with open(labels_path, 'r') as fin_json:
			labels_json = json.load(fin_json)
		labels = np.fromiter(labels_json.values(), dtype=float)
		assert labels.shape[0] == shape_withcolor.shape[0]
		assert np.amin(labels) >= 0
		assert np.amax(labels) <= 31

		# Export pointcloud for minkowski
		out_filepath = split_out_path / (shape + '.ply')
		processed = np.hstack((shape_withcolor, labels[:, np.newaxis]))
		save_point_cloud_with_normals(processed, out_filepath, with_label=True, verbose=False)





