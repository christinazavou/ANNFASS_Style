import os
import trimesh
import numpy as np

def read_obj(obj_fn):
	"""
		Read obj
	:param obj_fn: str
	:return:
		vertices: N x 3, numpy.ndarray(float)
    faces: M x 3, numpy.ndarray(int)
    components: M x 1, numpy.ndarray(int)
	"""

	assert (os.path.isfile(obj_fn))

	# Return variables
	vertices, faces, components = [], [], []

	with open(obj_fn, 'r') as f_obj:
		component = -1
		# Read obj geometry
		for line in f_obj:
			line = line.strip().split(' ')
			if line[0] == 'v':
				# Vertex row
				assert (len(line) == 4)
				vertex = [float(line[1]), float(line[2]), float(line[3])]
				vertices.append(vertex)
			if line[0] == 'o':
				# object row
				assert(len(line) == 2)
				component = int(line[1])
			if line[0] == 'f':
				# Face row
				face = [float(line[1].split('/')[0]), float(line[2].split('/')[0]), float(line[3].split('/')[0])]
				faces.append(face)
				components.append(component)

	vertices = np.vstack(vertices)
	faces = np.vstack(faces)
	components = np.vstack(components)

	return vertices, faces.astype(np.int32), components.astype(np.int32)


def read_ply(ply_fn):
	"""
		Read ply file
	:param ply_fn: str
	:return:
		vertices: N x 3, numpy.ndarray(float)
    faces: M x 3, numpy.ndarray(int)
	"""

	vertices, faces, n_vertices, n_faces = [], [], 0, 0
	header_end = False

	with open(ply_fn, 'r') as fin_ply:
		# Read header
		line = fin_ply.readline().strip()
		assert(line == "ply")
		for line in fin_ply:
			line = line.strip().split(' ')
			if line[0] == 	"end_header":
				# Header end
				header_end = True
				break
			if (line[0] == "element") and (line[1] == "vertex"):
				n_vertices = int(line[2])
			if (line[0] == "element") and (line[1] == "face"):
				n_faces = int(line[2])
		assert(header_end)

		# Read vertices
		for _ in range(n_vertices):
			line = fin_ply.readline().strip().split(' ')
			assert(len(line) >= 3)
			vertex = [float(line[0]), float(line[1]), float(line[2])]
			vertices.append(vertex)

		# Read faces
		for _ in range(n_faces):
			line = fin_ply.readline().strip().split(' ')
			assert(len(line) >= 4)
			n_face = int(line[0])
			face = []
			for line_idx in range(1, n_face+1):
				face.append(float(line[line_idx]))
			faces.append(face)

	if len(vertices):
		vertices = np.vstack(vertices)
	if len(faces):
		faces = np.vstack(faces)

	return vertices, faces


def write_ply(ply_fn, vertices, faces, face_color):
	"""
		Write shape in .ply with face color information
	:param ply_fn: str
	:param vertices: N x 3, numpy.ndarray(float)
	:param faces: M x 3, numpy.ndarray(int)
	:param face_color: M x 1, numpy.ndarray(float)
	:return:
		None
	"""

	# Create header
	header = 'ply\n' \
			 'format ascii 1.0\n' \
			 'element vertex ' + str(len(vertices)) + '\n' \
			 'property float x\n' \
			 'property float y\n' \
			 'property float z\n' \
			 'element face ' + str(len(faces)) + '\n' \
			 'property list uchar int vertex_indices\n' \
			 'property float red\n' \
			 'property float green\n' \
			 'property float blue\n' \
			 'end_header\n'

	if np.min(faces) == 1:
		faces -= 1

	with open(ply_fn, 'w') as f_ply:
		# Write header
		f_ply.write(header)

		# Write vertices
		for vertex in vertices:
			row = ' '.join([str(vertex[0]), str(vertex[1]), str(vertex[2])]) + '\n'
			f_ply.write(row)
		# Write faces + face_color
		for face_ind, face in enumerate(faces):
			color = face_color[face_ind]
			row = ' '.join([str(len(face)), str(face[0]), str(face[1]), str(face[2]),
							str(color[0]), str(color[1]), str(color[2])]) + '\n'
			f_ply.write(row)


def calculate_face_area(vertices, faces):
	""" Calculate face area of a triangular mesh
  :param vertices: N x 3, numpy.ndarray(float)
  :param faces: M x 3, numpy.ndarray(int)
  :return:
    face_area: M x 1, numpy.ndarray(float)
	"""

	# Get vertices of faces
	A = vertices[faces[:, 0]]
	B = vertices[faces[:, 1]]
	C = vertices[faces[:, 2]]

	# Create face edges
	e1 = B - A
	e2 = C - A

	# Calculate cross product and find length
	cross_prod = np.cross(e1, e2)
	cross_prod_len = np.sqrt(np.sum(cross_prod**2, axis=1))

	# Get face area
	face_area = cross_prod_len / 2.0

	return face_area[:, np.newaxis]


def sample_faces(vertices, faces, n_samples=100):
  """
    Samples point cloud on the surface of the model defined as vertices and
    faces. This function uses vectorized operations so fast at the cost of some
    memory.

    Parameters:
      vertices  - n x 3 matrix
      faces     - n x 3 matrix
      n_samples - positive integer

    Return:
      vertices - point cloud

    Reference :
      Barycentric coordinate system
        P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C

  """

  n_samples_per_face = np.zeros((len(faces),), dtype=int) + n_samples
  n_samples = np.sum(n_samples_per_face)

  # Create a vector that contains the face indices
  sample_face_idx = np.zeros((n_samples,), dtype=int)
  acc = 0
  for face_idx, _n_sample in enumerate(n_samples_per_face):
    sample_face_idx[acc: acc + _n_sample] = face_idx
    acc += _n_sample
  r = np.random.rand(n_samples, 2)
  A = vertices[faces[sample_face_idx, 0], :]
  B = vertices[faces[sample_face_idx, 1], :]
  C = vertices[faces[sample_face_idx, 2], :]
  P = (1 - np.sqrt(r[:, 0:1])) * A + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B + np.sqrt(r[:, 0:1]) * r[:, 1:] * C

  return P


if __name__ == "__main__":
  import json
  from collections import OrderedDict


  def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
      if len(scene_or_mesh.geometry) == 0:
        mesh = None  # empty scene
      else:
        # we lose texture information here
        mesh = trimesh.util.concatenate(
          tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in scene_or_mesh.geometry.values()))
    else:
      assert (isinstance(scene_or_mesh, trimesh.Trimesh))
      mesh = scene_or_mesh
    return mesh

  # BuildNet directories
  BUILDNET_BASE_DIR = os.path.join(os.sep, "media", "melinos", "BigData", "BUILDNET", "BUILDNET_2000")
  BUILDNET_OBJ_DIR = os.path.join(BUILDNET_BASE_DIR, "flippedNormal_unit_obj_withtexture")
  assert(os.path.isdir(BUILDNET_OBJ_DIR))
  BUILDNET_FACEAREA_DIR = os.path.join(BUILDNET_BASE_DIR, "facearea")
  assert(os.path.isdir(BUILDNET_FACEAREA_DIR))

  # Check face area
  mesh_list = [fn.split('.')[0] for fn in os.listdir(BUILDNET_FACEAREA_DIR)]
  c1, c2 = 0, 0
  for mesh_fn in mesh_list:
    input_obj_fn = os.path.join(BUILDNET_OBJ_DIR, mesh_fn+".obj")
    input_facearea_fn = os.path.join(BUILDNET_FACEAREA_DIR, mesh_fn+".json")
    # scene_or_mesh = trimesh.load(input_obj_fn, process=False, skip_material=True)
    # mesh = as_mesh(scene_or_mesh=scene_or_mesh)
    vertices, faces = read_obj(obj_fn=input_obj_fn)
    # assert(mesh.faces.shape == faces.shape)
    faces -= 1
    with open(input_facearea_fn, 'r') as fin_json:
      face_area_json = json.load(fin_json)
    face_area = np.fromiter(face_area_json.values(), dtype=np.float32)
    assert(faces.shape[0] == face_area.shape[0])
    calculated_face_area = calculate_face_area(vertices=vertices, faces=faces).astype(np.float32)
    # tri_face_area = mesh.area_faces
    # area_dict = OrderedDict(zip(np.arange(len(calculated_face_area)).astype('str'), calculated_face_area))
    # with open(mesh_fn+".json", 'w') as fout_json:
    #   json.dump(area_dict, fout_json)
    # try:
    #   # assert(np.isclose(tri_face_area, calculated_face_area).all())
    #   # assert(np.isclose(mesh.area, np.sum(calculated_face_area)))
    #   print("{} my face area calculation ok!!!" .format(mesh_fn))
    #   c1 += 1
    # except AssertionError:
    #   print("ERROR: {} my face area calculation not ok!!!".format(mesh_fn))
    try:
      # assert(np.isclose(tri_face_area, face_area).all())
      # assert (np.isclose(calculated_face_area, face_area).all())
      # assert(np.isclose(mesh.area, np.sum(face_area)))
      assert(np.isclose(np.sum(calculated_face_area), np.sum(face_area)))
      print("{} Pratheba face area calculation ok!!!".format(mesh_fn))
      c2 += 1
    except AssertionError:
      print("ERROR: {} Pratheba's face area calculation not ok!!!".format(mesh_fn))
  print("My method: correct ({}/{})" .format(c1, len(mesh_list)))
  print("Pratheba's method: correct ({}/{})".format(c2, len(mesh_list)))
