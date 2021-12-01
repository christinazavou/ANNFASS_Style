# Command line arguments
import argparse
import math
import os

import mathutils
import numpy as np
from plyfile import PlyElement, PlyData
from progressbar import ProgressBar
from scipy import spatial

import mp_utils

_THRESHOLD_TOL_32 = 2.0 * np.finfo(np.float32).eps
_THRESHOLD_TOL_64 = 2.0 * np.finfo(np.float32).eps

# Global variables
models_fns = []
models_data = {}
query_models_fns = []
query_models_data = {}


def pc_bounding_box_diagonal(pc):
	"""Calculate point cloud's bounding box"""
	centroid = np.mean(pc, axis=0)
	pc2 = pc - centroid

	# Calculate bounding box diagonal
	xyz_min = np.amin(pc2, axis=0)
	xyz_max = np.amax(pc2, axis=0)
	bb_diagonal = np.max([np.linalg.norm(xyz_max - xyz_min, ord=2), _THRESHOLD_TOL_64 if pc.dtype==np.float64 else _THRESHOLD_TOL_32])

	return bb_diagonal


def load_obj(filename, normalize=True):
	"""Load obj"""
	fin = open(filename, 'r')
	lines = [line.rstrip() for line in fin]
	fin.close()

	vertices = []; faces = []
	for line in lines:
		if line.startswith('v '):
			vertices.append(np.float32(line.split()[1:4]))
		elif line.startswith('f '):
			if len(line.split()) != 4:
				continue
			if "//" in line.split()[1]:
				faces.append(np.int32([item.split('//')[0] for item in line.split()[1:4]]))
			elif "/" in line.split()[1]:
				faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))
			else:
				faces.append(np.int32(line.split()[1:4]))

	if len(faces) > 0:
		f = np.vstack(faces)
		if np.amin(faces) == 1:
			f -= 1
	else:
		print(filename + " does not have any faces")
		f = np.array([])

	v = np.vstack(vertices)

	if normalize:
		# normalize diagonal=1
		x_max = np.max(v[:, 0])
		y_max = np.max(v[:, 1])
		z_max = np.max(v[:, 2])
		x_min = np.min(v[:, 0])
		y_min = np.min(v[:, 1])
		z_min = np.min(v[:, 2])
		x_mid = (x_max + x_min) / 2
		y_mid = (y_max + y_min) / 2
		z_mid = (z_max + z_min) / 2
		x_scale = x_max - x_min
		y_scale = y_max - y_min
		z_scale = z_max - z_min
		scale = np.sqrt(x_scale * x_scale + y_scale * y_scale + z_scale * z_scale)

		v[:, 0] = (v[:, 0] - x_mid) / scale
		v[:, 1] = (v[:, 1] - y_mid) / scale
		v[:, 2] = (v[:, 2] - z_mid) / scale

	return v, f


def check_for_duplicate(models_fns_sublist, proc_id):
	print("Begin process: {} with {} models_fns_sublist" .format(proc_id, len(models_fns_sublist)))

	# Check for duplicates
	distances = [];  model_source_ind = 0; model_target_ind = 0

	# Log current check
	current_file = os.path.join(RESULT_DIR, "log_current_"+str(proc_id)+".txt")

	# Duplicate tolerance
	# epsilon = 1e-6
	with open(os.path.join(RESULT_DIR, "distances_" + str(proc_id) + ".csv"), 'w') as fout:
		fout.write("source_models,target_models,equal_vertices,equal_faces,distances\n")

	bar = ProgressBar()
	print("Find duplicates within the dataset...")
	for i in bar(range(model_source_ind, len(query_models_fns))):
		model_source_fn = query_models_fns[i]
		source_vertices = query_models_data[model_source_fn]["vertices"]
		source_n_faces = query_models_data[model_source_fn]["n_faces"]

		source_target_dist = np.inf
		source_target_equal_v = False
		source_target_equal_f = False
		target_name = None

		# Measure chamfer distance from between models
		for j in range(model_target_ind, len(models_fns_sublist)):
			model_target_fn = models_fns_sublist[j]
			if model_source_fn == model_target_fn:
				continue
			with open(current_file, 'w') as fout:
				fout.write(str(i)+","+str(j)+": "+model_source_fn + " -> " + model_target_fn)
			target_vertices = models_data[model_target_fn]["vertices"]
			target_n_faces = models_data[model_target_fn]["n_faces"]

			for rotation in range(360 // ARGS.rotation):
				# rotate source_vertices

				rot = mathutils.Matrix.Rotation(math.radians(ARGS.rotation * rotation), 3, (0, 1, 0))
				source_vertices_rot = np.matmul(source_vertices, rot)

				source_KD_Tree = spatial.cKDTree(source_vertices_rot, copy_data=False, balanced_tree=False,
												 compact_nodes=False)

				for k in range(2):
					if k == 0:
						v1 = source_vertices_rot; v2 = target_vertices
						# Create kd-tree for v2
						v2_KD_Tree = spatial.cKDTree(v2, copy_data=False, balanced_tree=False,
													 compact_nodes=False)
					if k == 1:
						v1 = target_vertices; v2 = source_vertices_rot
						# Assign source_KD_Tree to v2_KD_Tree
						v2_KD_Tree = source_KD_Tree

					# Find v2 points 1-nn for each point in v1
					nn_dist, _ = v2_KD_Tree.query(v1, k=1)
					directional_mean_dist = np.mean(nn_dist) / pc_bounding_box_diagonal(v1)

					if directional_mean_dist < source_target_dist:
						source_target_dist = directional_mean_dist
						source_target_equal_v = source_target_equal_v or len(source_vertices_rot) == len(target_vertices)
						source_target_equal_f = source_target_equal_f or source_n_faces == target_n_faces
						target_name = model_target_fn

		distances.append((
			model_source_fn,
			target_name,
			source_target_equal_v,
			source_target_equal_f,
			source_target_dist
		))
		if i % 10 == 0 or i == len(query_models_fns) - 1:
			with open(os.path.join(RESULT_DIR, "distances_"+str(proc_id)+".csv"), 'a') as fout:
				for dist in distances:
					fout.write(",".join([str(x) for x in dist]) + "\n")
				distances = []
	print("Terminate process: {proc_id:d}".format(proc_id=proc_id))


def write_ply_v_f(vertices, faces, out_file):
	vertices = np.array([tuple(v) for v in vertices], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
	faces = np.array([(list(f),) for f in faces], dtype=[('vertex_indices', 'i4', (3,))])
	elv = PlyElement.describe(vertices, 'vertex')
	elf = PlyElement.describe(faces, 'face')
	PlyData([elv, elf], text=True).write(out_file)


if __name__ == "__main__":
	models_n_faces = []

	parser = argparse.ArgumentParser()
	parser.add_argument("--n_proc", type=int, default=1, help="Number of processes to use [default: 1]")
	parser.add_argument("--start_ind", type=int, default=0, help="StartIdx [default: 0]")
	parser.add_argument("--end_ind", type=int, default=100, help="EndIdx [default: 1]")
	parser.add_argument("--big_dir", type=str, help="big_dir", required=True)
	parser.add_argument("--query_dir", type=str, help="query_dir", required=True)
	parser.add_argument("--filenames", type=str, help="filenames", required=True)
	parser.add_argument("--result_dir", type=str, help="result_dir")
	parser.add_argument("--rotation", type=int, help="angle to rotate default is 360 i.e. none", default=360)
	parser.add_argument("--ref_model_name", type=str, default="model.obj")
	parser.add_argument("--query_model_name", type=str, default="whole/model.obj")
	ARGS = parser.parse_args()
	print(ARGS)

	QUERY_DIR = ARGS.query_dir
	ABC_DIR = ARGS.big_dir
	FILENAMES_FILE = ARGS.filenames
	if not ARGS.result_dir:
		RESULT_DIR = os.path.curdir
	else:
		RESULT_DIR = ARGS.result_dir
	assert os.path.exists(QUERY_DIR)
	assert os.path.exists(ABC_DIR)
	assert os.path.exists(FILENAMES_FILE)
	if not os.path.exists(RESULT_DIR):
		os.makedirs(RESULT_DIR)

	n_proc = ARGS.n_proc

	# Read models filenames
	with open(FILENAMES_FILE, 'r') as fin:
		for line in fin:
			models_fns.append(line.strip())
	# models_fns = models_fns[0:5]

	query_models_fns = [f for f in os.listdir(QUERY_DIR)
						if os.path.isfile(os.path.join(QUERY_DIR, f, ARGS.query_model_name))]
	# query_models_fns = query_models_fns[0:5]

	# Load all models
	print("Load big dir models...")
	bar = ProgressBar()
	for i in bar(range(len(models_fns))):
		# Load obj
		model_fn = models_fns[i]
		model_obj = os.path.join(ABC_DIR, model_fn, "{}.obj".format(model_fn))
		if not os.path.exists(model_obj):
			model_obj = os.path.join(ABC_DIR, model_fn, ARGS.ref_model_name)
			assert os.path.exists(model_obj), "{} (2) doesnt exist".format(model_obj)
		V_model, F_model = load_obj(filename=model_obj, normalize=True)
		models_data[model_fn] = {"vertices": V_model, "n_faces": len(F_model)}
		# models_data[model_fn] = {"vertices": V_model, "n_faces": len(F_model), "faces": F_model}

	# Load all models
	print("Load query dir models...")
	bar = ProgressBar()
	for i in bar(range(len(query_models_fns))):
		# Load obj
		model_fn = query_models_fns[i]
		model_obj = os.path.join(QUERY_DIR, model_fn, ARGS.query_model_name)
		V_model, F_model = load_obj(filename=model_obj, normalize=True)
		query_models_data[model_fn] = {"vertices": V_model, "n_faces": len(F_model)}
		# query_models_data[model_fn] = {"vertices": V_model, "n_faces": len(F_model), "faces": F_model}

	if ARGS.end_ind > len(models_fns):
		ARGS.end_ind = len(models_fns)
	# check_for_duplicate(models_fns[ARGS.start_ind:ARGS.end_ind], 0)
	mp_utils.runParallelFunc(check_for_duplicate, n_proc, models_fns[ARGS.start_ind:ARGS.end_ind])
	print("Finished all processes")

	# Merge .csv
	overall_distances = {}
	for f in os.listdir(RESULT_DIR):
		if f.endswith(".csv"):
			with open(os.path.join(RESULT_DIR, f), 'r') as fin:
				line = fin.readline().strip().split(',')
				assert(line[0] == "source_models")
				assert(line[1] == "target_models")
				assert(line[2] == "equal_vertices")
				assert(line[3] == "equal_faces")
				assert(line[4] == "distances")
				for line in fin:
					line = line.strip().split(',')
					if line[0] not in overall_distances:
						overall_distances[line[0]] = (line[1], line[2], line[3], line[4])
					elif line[4] < overall_distances[line[0]][3]:
						overall_distances[line[0]] = (line[1], line[2], line[3], line[4])
	with open(os.path.join(RESULT_DIR, "distances.csv"), 'w') as fout:
		# Write header
		fout.write("source_models,target_models,equal_vertices,equal_faces,distances\n")
		for query_name, distance in overall_distances.items():
			fout.write("{} {} {} {} {}\n".format(query_name,
												 distance[0],
												 distance[1],
												 distance[2],
												 distance[3]))
			# r_data = models_data[row[1]]
			# q_data = query_models_data[row[0]]
			# write_ply_v_f(r_data['vertices'], r_data['faces'], row[1]+".ply")
			# write_ply_v_f(q_data['vertices'], q_data['faces'], row[0]+".ply")
