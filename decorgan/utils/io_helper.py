import numpy as np
import os
import logging


def parse_simple_obj_file(obj_file):
    with open(obj_file, "r") as fin:
        lines = fin.readlines()
    vertices = []
    faces = []
    for line in lines:
        if line.startswith("v "):
            vertices.append(np.asarray(line[2:-1].split(" ")).astype(float))
        else:
            faces.append(np.asarray(line[2:-1].split(" ")).astype(float))
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if np.min(faces) == 1:
        faces = faces - 1
    return vertices, faces


def parse_xyz_rgb_ply(points_file):
    xyzs = []
    rgbs = []
    point_cnt = 0
    with open(points_file, "r") as fin:
        header = True
        for line in fin:
            if line[0].isdigit() or line[0] == '-':
                header = False
            if header:
                continue
            line = line.strip()
            data = line.split()
            sampled_point_coords = np.array([float(data[0]), float(data[1]), float(data[2])])
            xyzs.append(sampled_point_coords)
            sampled_point_colours = np.array([float(data[3]), float(data[4]), float(data[5])])
            rgbs.append(sampled_point_colours)
            point_cnt += 1
    print("Loaded points count: {}".format(len(xyzs)))
    return np.vstack(xyzs), np.vstack(rgbs)


class Vertices(list):
    def __init__(self):
        super().__init__()


class Face:

    def __init__(self, v_indices, vn_indices, vt_indices, material, component):
        self.v_indices = v_indices
        self.vn_indices = vn_indices
        self.vt_indices = vt_indices
        self.material = material
        self.component = component


class ObjMeshBasic:

    def _add_vertex(self, line, vertices):
        assert (len(line) == 4)
        vertex = np.array([float(line[1]), float(line[2]), float(line[3])])
        vertices.append(vertex)

    def _new_face_without_uv(self, line, material_name, component_id):
        v_vn_1, v_vn_2, v_vn_3 = line[1].split('//'), line[2].split('//'), line[3].split('//')
        assert len(v_vn_1) == 2 and len(v_vn_2) == 2 and len(v_vn_3) == 2

        v_idx_1, v_idx_2, v_idx_3 = int(v_vn_1[0]) - 1, int(v_vn_2[0]) - 1, int(v_vn_3[0]) - 1
        vn_idx_1, vn_idx_2, vn_idx_3 = int(v_vn_1[1]) - 1, int(v_vn_2[1]) - 1, int(v_vn_3[1]) - 1

        return Face([v_idx_1, v_idx_2, v_idx_3],
                    [vn_idx_1, vn_idx_2, vn_idx_3],
                    None,
                    material_name,
                    component_id)

    def _new_face_with_uv(self, line, material_name, component_id):
        v_uv_vn_1, v_uv_vn_2, v_uv_vn_3 = line[1].split('/'), line[2].split('/'), line[3].split('/')
        assert len(v_uv_vn_1) == 3 and len(v_uv_vn_2) == 3 and len(v_uv_vn_3) == 3

        v_idx_1, v_idx_2, v_idx_3 = int(v_uv_vn_1[0]) - 1, int(v_uv_vn_2[0]) - 1, int(v_uv_vn_3[0]) - 1
        uv_idx_1, uv_idx_2, uv_idx_3 = int(v_uv_vn_1[1]) - 1, int(v_uv_vn_2[1]) - 1, int(
            v_uv_vn_3[1]) - 1
        vn_idx_1, vn_idx_2, vn_idx_3 = int(v_uv_vn_1[2]) - 1, int(v_uv_vn_2[2]) - 1, int(
            v_uv_vn_3[2]) - 1

        return Face([v_idx_1, v_idx_2, v_idx_3],
                    [vn_idx_1, vn_idx_2, vn_idx_3],
                    [uv_idx_1, uv_idx_2, uv_idx_3],
                    material_name,
                    component_id), (uv_idx_1, uv_idx_2, uv_idx_3)

    def _add_face(self, line, faces, material_name, component_id):
        if len(line) < 4 or len(line) > 4:
            raise Exception("face has more/less than 3 vertices")

        if "//" in line[1]:
            faces.append(self._new_face_without_uv(line, material_name, component_id))
        else:
            faces.append(self._new_face_with_uv(line, material_name, component_id)[0])

    def __init__(self, obj_fn):
        self._load(obj_fn)

    def _load(self, obj_fn):

        assert (os.path.isfile(obj_fn))
        obj_dir = os.path.dirname(obj_fn)

        # Return variables
        vertices = Vertices()
        faces = []

        with open(obj_fn, 'r') as f_obj:

            # Read obj geometry
            line_idx = -1  # for debugging
            for line in f_obj:
                line_idx += 1
                line = line.strip().split(' ')

                if line[0] == 'v':
                    self._add_vertex(line, vertices)

                elif line[0] == 'f':
                    self._add_face(line, faces, None, None)

        self.vertex_coords = np.vstack(vertices)
        self.faces = np.vstack([face.v_indices for face in faces])

    def __str__(self):
        return "ObjMesh: {} vertices, {} faces\n".format(
            len(self.vertex_coords), len(self.faces))


def setup_logging(log_dir, filename="log.txt"):
    os.makedirs(log_dir, exist_ok=True)

    logpath = os.path.join(log_dir, filename)
    filemode = 'a' if os.path.exists(logpath) else 'w'

    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=logpath,
                        filemode=filemode)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
