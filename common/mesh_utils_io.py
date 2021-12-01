import os
import re
from typing import Union

import PIL
from PIL import Image
from plyfile import PlyData, PlyElement
from psd_tools import PSDImage

from common import *

PIL.Image.MAX_IMAGE_PIXELS=None


class SampledPoints:

    def __call__(self, points_file, face_index_file, faces=None):
        """
        :param points_file: either .pts or .ply
        """
        face_indices = []
        with open(os.path.join(face_index_file), "r") as fin:
            for index in fin:
                index = int(index.strip())
                if faces:
                    assert index < len(faces), "face index can't be that big :/"
                face_indices.append(index)

        points = []
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
                sampled_point_normals = np.array([float(data[3]), float(data[4]), float(data[5])])
                face_idx = face_indices[point_cnt]
                point_cnt += 1
                points.append(SampledPoint(sampled_point_coords, sampled_point_normals, face_idx))
        print("Loaded points count: {}".format(len(points)))
        return points


class SampledPointsWithColor:

    def __call__(self, points_file, face_index_file, faces=None):
        """
        :param points_file: either .pts or .ply
        """
        face_indices = []
        with open(os.path.join(face_index_file), "r") as fin:
            for index in fin:
                index = int(index.strip())
                if faces:
                    assert index < len(faces), "face index can't be that big :/"
                face_indices.append(index)

        points = []
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
                sampled_point_colours = np.array([int(data[3]), int(data[4]), int(data[5])])
                face_idx = face_indices[point_cnt]
                point_cnt += 1
                points.append(SampledPointWithColor(sampled_point_coords, face_idx, None, sampled_point_colours))
        print("Loaded points count: {}".format(len(points)))
        return points


class FullSampledPoints:

    def __call__(self, points_file, face_index_file,
                 ridge_valley_file=None, color_file=None, faces=None):
        """
        :param points_file: either .pts or .ply
        """
        face_indices = []
        with open(os.path.join(face_index_file), "r") as fin:
            for index in fin:
                index = int(index.strip())
                if faces:
                    assert index < len(faces), "face index can't be that big :/"
                face_indices.append(index)

        rnvs = []
        if ridge_valley_file is not None:
            with open(os.path.join(ridge_valley_file), "r") as fin:
                for rnv in fin:
                    rnv = int(rnv.strip())
                    rnvs.append(rnv)

        colors = []
        if color_file is not None:
            with open(color_file, "r") as fin:
                header = True
                for line in fin:
                    if line[0].isdigit() or line[0] == '-':
                        header = False
                    if header:
                        continue
                    line = line.strip()
                    data = line.split()
                    colors.append(np.array(data[3:6]).astype(np.uint8))

        points = []
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
                # sampled_point_coords = np.array(data[0:3]).astype(float)
                # sampled_point_normals = np.array(data[3:6]).astype(float)
                # sampled_point_pca = np.array(data[6:24]).astype(float)
                # sampled_point_curvature = np.array(data[24:64]).astype(float)
                points.append(np.array(data).astype(float))
                # face_idx = face_indices[point_cnt]
                point_cnt += 1
                # points.append(SampledPoint(sampled_point_coords, sampled_point_normals, face_idx))
        points = np.vstack(points)
        face_indices = np.array(face_indices)
        rnvs = np.array(rnvs, dtype=np.int8)
        colors = np.array(colors, dtype=np.uint8)
        print("Loaded points count: {}".format(len(points)))
        if color_file is not None and len(colors) < len(points):
            len_colors = len(colors)
            points = points[:len_colors]
            face_indices = face_indices[:len_colors]
            rnvs = rnvs[:len_colors]
            colors = colors[:len_colors]
            print("Kept points count: {}".format(len(points)))
        return points, face_indices, rnvs, colors


def read_pts(points_file):
    coords = []
    normals = []
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
            coords.append(sampled_point_coords)
            if len(data) >= 6:
                sampled_point_normals = np.array([float(data[3]), float(data[4]), float(data[5])])
                normals.append(sampled_point_normals)
            point_cnt += 1
    print("Loaded points count: {}".format(len(coords)))
    if len(normals) > 0:
        return np.vstack(coords), np.vstack(normals)
    return np.vstack(coords)


def read_face_indices(face_index_file):
    face_indices = []
    with open(os.path.join(face_index_file), "r") as fin:
        for index in fin:
            index = int(index.strip())
            face_indices.append(index)
    print("Loaded face indices count: {}".format(len(face_indices)))
    return face_indices


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


class ObjMeshBasic:

    def _add_vertex(self, line, vertices):
        assert (len(line) == 4)
        vertex = np.array([float(line[1]), float(line[2]), float(line[3])])
        vertices.append(vertex)

    def _add_vertex_normal(self, line, vertex_normals):
        assert (len(line) == 4)
        vertex_normal = np.array([float(line[1]), float(line[2]), float(line[3])])
        vertex_normals.append(vertex_normal)

    def _add_texture_coord(self, line, texture_coords):
        assert (len(line) == 3)
        texture_coords.append(np.array([float(line[1]), float(line[2])]))

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
        vertex_normals = VertexNormals()
        texture_coords = TextureCoords()
        faces = []

        with open(obj_fn, 'r') as f_obj:

            # Read obj geometry
            line_idx = -1  # for debugging
            for line in f_obj:
                line_idx += 1
                line = line.strip().split(' ')

                if line[0] == 'v':
                    self._add_vertex(line, vertices)

                elif line[0] == 'vn':
                    self._add_vertex_normal(line, vertex_normals)

                elif line[0] == "vt":
                    self._add_texture_coord(line, texture_coords)

                elif line[0] == 'f':
                    self._add_face(line, faces, None, None)

        self.vertex_coords = np.vstack(vertices)
        if len(vertex_normals) > 0:
            self.vertex_normals = np.vstack(vertex_normals)
        if len(texture_coords) > 0:
            self.texture_coords = np.vstack(texture_coords)
        self.faces = faces

    def __str__(self):
        return "ObjMesh: {} vertices, {} uvs, {} faces\n".format(
            len(self.vertex_coords), len(self.texture_coords), len(self.faces))


class ObjMeshWithComponents(ObjMeshBasic):

    def __init__(self, obj_fn):
        super(ObjMeshWithComponents, self).__init__(obj_fn)

    def _load(self, obj_fn):
        assert (os.path.isfile(obj_fn)), f"{obj_fn} is not an existing file path"

        # Return variables
        vertices = Vertices()
        vertex_normals = VertexNormals()
        texture_coords = TextureCoords()
        faces = []
        components = []

        current_mat = None

        component_cnt = -1

        with open(obj_fn, 'r') as f_obj:

            # Read obj geometry
            line_idx = -1  # for debugging
            for line in f_obj:
                line_idx += 1
                line = line.strip().split(' ')

                if line[0] == 'v':
                    self._add_vertex(line, vertices)

                elif line[0] == 'vn':
                    self._add_vertex_normal(line, vertex_normals)

                elif line[0] == 'o':
                    # object row
                    assert len(line) == 2
                    component_cnt += 1
                    if line[1] != "":
                        component_id = line[1]
                    else:
                        component_id = str(component_cnt)
                    components.append(component_id)

                elif line[0] == "usemtl":
                    current_mat = " ".join(line[1:]).strip()

                elif line[0] == "vt":
                    self._add_texture_coord(line, texture_coords)

                elif line[0] == 'f':
                    # Face row

                    if len(line) < 4 or len(line) > 4:
                        raise Exception("face has more/less than 3 vertices")

                    if "//" in line[1]:
                        faces.append(self._new_face_without_uv(line, current_mat, component_id))
                    else:
                        new_face, _ = self._new_face_with_uv(line, current_mat, component_id)
                        faces.append(new_face)

        self.vertex_coords = np.vstack(vertices)
        if len(vertices) > 0:
            self.vertex_normals = np.vstack(vertex_normals)
        if len(texture_coords) > 0:
            self.texture_coords = np.vstack(texture_coords)
        self.faces = faces
        self.components = components

    def __str__(self):
        return "ObjMesh: {} vertices, {} uvs, {} faces, {} components\n".format(
            len(self.vertex_coords), len(self.texture_coords), len(self.faces), len(self.components))


class ObjMeshComponentsReference(ObjMeshBasic):

    def __init__(self, obj_fn):
        super(ObjMeshComponentsReference, self).__init__(obj_fn)

    def _load(self, obj_fn):
        assert (os.path.isfile(obj_fn))

        # Return variables
        faces = []
        components = []

        current_mat = None

        component_cnt = -1

        with open(obj_fn, 'r') as f_obj:

            # Read obj geometry
            line_idx = -1  # for debugging
            for line in f_obj:
                line_idx += 1
                line = line.strip().split(' ')

                if line[0] == 'o':
                    # object row
                    assert len(line) == 2
                    component_cnt += 1
                    if line[1] != "":
                        component_id = line[1]
                    else:
                        component_id = str(component_cnt)
                    components.append(component_id)

                elif line[0] == 'f':
                    # Face row

                    if len(line) < 4 or len(line) > 4:
                        raise Exception("face has more/less than 3 vertices")

                    if "//" in line[1]:
                        faces.append(self._new_face_without_uv(line, current_mat, component_id))
                    else:
                        new_face, _ = self._new_face_with_uv(line, current_mat, component_id)
                        faces.append(new_face)

        self.faces = faces
        self.components = components

    def __str__(self):
        return "{}: {} faces, {} components\n".format(self.__class__.__name__, len(self.faces), len(self.components))


class Material:
    def __init__(self, name):
        self.kd = {"r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0}
        self.d = 1.0  # if 0 it means completely transparent.
        self.name = name
        self.bump_img_file = None

    def set_rgb_diffuse(self, r, g, b, a=None):
        assert 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1, "Not recognized kd values"
        self.kd["r"] = r
        self.kd["g"] = g
        self.kd["b"] = b
        if a:
            self.kd["a"] = a

    def set_bump_img_file(self, bump_img_file):
        self.bump_img_file = bump_img_file

    def __str__(self):
        return "(Material) {}".format(self.name)


class MaterialWithTexture(Material):

    def __init__(self, material, img_file):
        super(MaterialWithTexture, self).__init__(material.name)
        self.name = material.name
        self.kd = material.kd
        self.d = material.d
        self.pix = np.array(self.load_image(img_file))  # Y, X, 3 (i.e. accessing v, u)
        self.w = self.pix.shape[1]
        self.h = self.pix.shape[0]
        self.bump_img_file = material.bump_img_file
        self.img_file = img_file
        self.scale_x = 1.
        self.scale_y = 1.

        # # debugging:
        # if self.bump_img_file != None:
        #     bump_pix = np.array(self.load_image(self.bump_img_file))
        #     print(f"Bump img same size? {bump_pix.shape == self.pix.shape}")

    def set_scale(self, scale_x, scale_y):
        self.scale_x = scale_x
        self.scale_y = scale_y

    @staticmethod
    def load_image(imgfilepath):
        fname, ext = os.path.splitext(imgfilepath)
        if ext == '.psd':
            img = PSDImage.open(imgfilepath).topil()
        else:
            img = Image.open(imgfilepath)

        # ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = img.convert('RGB')

        # img = img.filter(ImageFilter.BLUR)
        return img

    def __str__(self):
        return "(MaterialWithTexture) {} (w,h: {}, {})".format(self.name, self.w, self.h)


class MTLMaterials(dict):

    def __init__(self, mtl_file):
        super().__init__()

        mtl_dir = os.path.dirname(mtl_file)
        fin = open(mtl_file, 'r')

        current_mat_name = None
        for line in fin:
            line = line.strip()

            if len(line):
                if line.startswith("newmtl "):
                    current_mat_name = line.strip().replace("newmtl ", "")
                    current_material = Material(current_mat_name)
                    self.__setitem__(current_mat_name, current_material)
                if line.startswith("Kd "):
                    data = line.split()
                    if len(data) == 5:
                        upd = self.__getitem__(current_mat_name)
                        upd.set_rgb_diffuse(float(data[1]), float(data[2]), float(data[3]), float(data[4]))
                        self.__setitem__(current_mat_name, upd)
                    else:
                        upd = self.__getitem__(current_mat_name)
                        upd.set_rgb_diffuse(float(data[1]), float(data[2]), float(data[3]))
                        self.__setitem__(current_mat_name, upd)
                if line.startswith("d "):
                    upd = self.__getitem__(current_mat_name)
                    upd.d = float(line.split()[1])
                    self.__setitem__(current_mat_name, upd)
                if line.startswith("map_Bump "):
                    upd = self.__getitem__(current_mat_name)
                    bump_img_file = line.strip().split(" ")[-1]
                    upd.set_bump_img_file(os.path.join(mtl_dir, bump_img_file))
                    self.__setitem__(current_mat_name, upd)
                if line.startswith("map_Kd "):
                    match_scale = re.compile(r"(-s) (\d+\.\d+) (\d+\.\d+) (\d+\.\d+) .", re.UNICODE).findall(line)
                    match_offset = re.compile(r"(-o) ([\-\d]+\.\d+) ([\-\d]+\.\d+) ([\-\d]+\.\d+) .", re.UNICODE).findall(line)
                    the_line = line
                    scale = np.array([1., 1., 1.])
                    if len(match_scale) > 0:
                        scale = np.array(match_scale[0][1:4], dtype=float)
                        replace = " ".join(match_scale[0])
                        the_line = the_line.replace(replace, "")
                    if len(match_offset) > 0:
                        replace = " ".join(match_offset[0])
                        the_line = the_line.replace(replace, "")
                    img_file = the_line.strip().split("map_Kd ")[-1].strip()
                    if os.path.exists(os.path.join(mtl_dir, img_file)):
                        self.__setitem__(current_mat_name, MaterialWithTexture(self.__getitem__(current_mat_name),
                                                                               os.path.join(mtl_dir, img_file)))
                        upd = self.__getitem__(current_mat_name)
                        upd.set_scale(scale[0], scale[1])
                        self.__setitem__(current_mat_name, upd)
                    else:
                        print(f"Warning: File {os.path.join(mtl_dir, img_file)} doesnt exist")

        fin.close()

    def __str__(self):
        return "MTLMaterials: " + ", ".join(
            ["{}: {}".format(key, value) for key, value in self.items() if key is not None])


class ObjMeshWithComponentsAndMaterials(ObjMeshBasic):

    def __init__(self, obj_fn):
        self.min_max_uv_per_material_per_component = dict()
        super(ObjMeshWithComponentsAndMaterials, self).__init__(obj_fn)

    def _load(self, obj_fn):
        assert (os.path.isfile(obj_fn))
        obj_dir = os.path.dirname(obj_fn)

        # Return variables
        vertices = Vertices()
        vertex_normals = VertexNormals()
        texture_coords = TextureCoords()
        faces = []
        components = []

        current_mat = None

        component_cnt = -1
        component_id = str(component_cnt)

        with open(obj_fn, 'r') as f_obj:

            # Read obj geometry
            line_idx = -1  # for debugging
            for line in f_obj:
                line_idx += 1
                line = line.strip().split(' ')

                if line[0] == "mtllib":
                    materials = MTLMaterials(os.path.join(obj_dir, " ".join(line[1:]).strip()))

                elif line[0] == 'v':
                    self._add_vertex(line, vertices)

                elif line[0] == 'vn':
                    self._add_vertex_normal(line, vertex_normals)

                elif line[0] == 'o':
                    # object row
                    assert len(line) == 2
                    component_cnt += 1
                    if line[1] != "":
                        component_id = line[1]
                    else:
                        component_id = str(component_cnt)
                    if component_id in components:
                        print(f"WARNING: component {component_id} already mentioned in .obj file.")
                    components.append(component_id)

                elif line[0] == "usemtl":
                    current_mat = " ".join(line[1:]).strip()
                    if materials:
                        assert current_mat in materials

                elif line[0] == "vt":
                    self._add_texture_coord(line, texture_coords)

                elif line[0] == 'f':
                    # Face row

                    if len(line) < 4:
                        raise Exception("face has less than 3 vertices")

                    if len(line) > 4:
                        raise Exception("face has more than 3 vertices")

                    if "//" in line[1]:
                        faces.append(self._new_face_without_uv(line, current_mat, component_id))
                        if materials:
                            assert not isinstance(materials[current_mat], MaterialWithTexture)
                    else:
                        new_face, uv_indices = self._new_face_with_uv(line, current_mat, component_id)
                        faces.append(new_face)

        self.vertex_coords = np.vstack(vertices)
        self.vertex_normals = np.vstack(vertex_normals)
        self.texture_coords = np.vstack(texture_coords)
        self.faces = faces
        self.components = components
        self.materials = materials if materials else {}

        self._find_min_max_uv_per_material_per_component()

        print("WARNING: material-component pairs with un-normalized uvs:")
        for material, material_dict in self.min_max_uv_per_material_per_component.items():
            for component, component_dict in material_dict.items():
                uv_bounds = np.array(list(component_dict.values()))
                if not all(np.logical_and(0 <= uv_bounds, uv_bounds <= 1)):
                    print(f"{material}, {component}")

    def _find_min_max_uv_per_material_per_component(self):
        min_max_dict = self.min_max_uv_per_material_per_component
        for face in self.faces:
            if face.vt_indices:
                min_max_dict.setdefault(face.material, dict())
                min_max_dict[face.material].setdefault(face.component,
                                                       {'min_u': 1e9, 'min_v': 1e9, 'max_u': 1e-9, 'max_v': 1e-9})
                cur_dict = min_max_dict[face.material][face.component]
                for uv in face.vt_indices:
                    uv = self.texture_coords[uv]
                    cur_dict['min_u'] = min(uv[0], cur_dict['min_u'])
                    cur_dict['min_v'] = min(uv[1], cur_dict['min_v'])
                    cur_dict['max_u'] = max(uv[0], cur_dict['max_u'])
                    cur_dict['max_v'] = max(uv[1], cur_dict['max_v'])
        self.min_max_uv_per_material_per_component = min_max_dict

    def __str__(self):
        return "ObjMesh: {} vertices, {} uvs, {} faces, {} components and materials:\n\t\t{}\n".format(
            len(self.vertex_coords), len(self.texture_coords), len(self.faces), len(self.components), self.materials)



def export_selection_obj(in_file, out_file, substrings=None):
    assert (os.path.isfile(in_file))

    include_components_faces = True if substrings is None else False

    with open(in_file, 'r') as f_in, open(out_file, "w") as f_out:

        # Read obj geometry
        line_idx = -1  # for debugging
        for line in f_in:
            line_idx += 1

            if line.startswith('o '):
                # object row
                component_id = line.split(' ')[1]
                if substrings is None or any(s for s in substrings if s.lower() in component_id.lower()):
                    include_components_faces = True
                    f_out.write(line)
                else:
                    include_components_faces = False

            elif line[0] == 'f' or line[0] == "l":
                if include_components_faces:
                    f_out.write(line)
            else:
                f_out.write(line)


class MeshWithAdjacency:
    def __init__(self):
        self.vertices = np.array([], dtype=np.float)
        self.faces = []
        self.vertices_neighbours = dict()


class ObjMeshWithAdjacency(ObjMeshBasic, MeshWithAdjacency):

    def __init__(self, obj_fn):
        super(ObjMeshWithAdjacency, self).__init__(obj_fn)

    def _load(self, obj_fn):
        assert (os.path.isfile(obj_fn))

        # Return variables
        vertices = Vertices()
        vertices_neighbours = {}

        with open(obj_fn, 'r') as f_obj:

            # Read obj geometry
            line_idx = -1  # for debugging
            for line in f_obj:
                line_idx += 1
                line = line.strip().split(' ')

                if line[0] == 'v':
                    self._add_vertex(line, vertices)
                    vertices_neighbours.setdefault(len(vertices) - 1, [])

                elif line[0] == 'f':
                    # Face row

                    if len(line) < 4 or len(line) > 4:
                        raise Exception("face has more/less than 3 vertices")

                    if "//" in line[1]:
                        new_face = self._new_face_without_uv(line, None, None)
                    else:
                        new_face, _ = self._new_face_with_uv(line, None, None)

                    vertices_neighbours[new_face.v_indices[0]].append(new_face.v_indices[1])
                    vertices_neighbours[new_face.v_indices[0]].append(new_face.v_indices[2])
                    vertices_neighbours.setdefault(new_face.v_indices[1], [])
                    vertices_neighbours[new_face.v_indices[1]].append(new_face.v_indices[0])
                    vertices_neighbours[new_face.v_indices[1]].append(new_face.v_indices[2])
                    vertices_neighbours.setdefault(new_face.v_indices[2], [])
                    vertices_neighbours[new_face.v_indices[2]].append(new_face.v_indices[0])
                    vertices_neighbours[new_face.v_indices[2]].append(new_face.v_indices[1])

        self.vertices = np.array(vertices)
        # remove redundancy:
        for v_idx in vertices_neighbours:
            vertices_neighbours[v_idx] = list(set(vertices_neighbours[v_idx]))
        self.vertices_neighbours = vertices_neighbours

    def __str__(self):
        return "{}: {} vertices\n".format(self.__class__.__name__, len(self.vertices))


class PlyWithAdjacency(MeshWithAdjacency):

    def __init__(self, obj_fn):
        super(PlyWithAdjacency, self).__init__()

        plydata = PlyData.read(obj_fn)
        vertices = plydata.elements[0].data
        faces = plydata.elements[1].data
        self.faces = np.array([f[0] for f in faces]).astype(int)
        coords = np.array([vertices['x'], vertices['y'], vertices['z']], dtype=np.float32).T
        vertices_neighbours = {}
        self.vertices = coords
        for idx, vertex in enumerate(self.vertices):
            vertices_neighbours.setdefault(idx, [])

        for face_v_indices in faces:
            face_v_indices = face_v_indices[0]
            assert len(face_v_indices) == 3, "Face is not a triangle"
            vertices_neighbours.setdefault(face_v_indices[0], [])
            vertices_neighbours[face_v_indices[0]].append(face_v_indices[1])
            vertices_neighbours[face_v_indices[0]].append(face_v_indices[2])
            vertices_neighbours.setdefault(face_v_indices[1], [])
            vertices_neighbours[face_v_indices[1]].append(face_v_indices[0])
            vertices_neighbours[face_v_indices[1]].append(face_v_indices[2])
            vertices_neighbours.setdefault(face_v_indices[2], [])
            vertices_neighbours[face_v_indices[2]].append(face_v_indices[0])
            vertices_neighbours[face_v_indices[2]].append(face_v_indices[1])
        self.vertices_neighbours = vertices_neighbours


def write_ply_v_f(vertices: Union[np.ndarray, list], faces: Union[np.ndarray, list], out_file):
    vertices = np.array([tuple(v) for v in vertices], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    faces = np.array([(list(f),) for f in faces], dtype=[('vertex_indices', 'i4', (3,))])
    elv = PlyElement.describe(vertices, 'vertex')
    elf = PlyElement.describe(faces, 'face')
    PlyData([elv, elf], text=True).write(out_file)


def read_ply_data(ply_file):
    plydata = PlyData.read(ply_file)
    vertices = plydata.elements[0].data
    faces = plydata.elements[1].data
    faces = np.array([f[0] for f in faces]).astype(int)
    vertices = np.array([vertices['x'], vertices['y'], vertices['z']], dtype=np.float32).T
    return vertices, faces


def read_ply(ply_fn):
    vertices, n_vertices = [], 0
    header_end = False

    with open(ply_fn, 'r') as fin_ply:
        # Read header
        with_label = False
        with_normal = False
        line = fin_ply.readline().strip()
        assert (line == "ply")
        lines = fin_ply.readlines()
        for lineidx, line in enumerate(lines):
            line = line.strip().split(' ')
            if line[0] == "end_header":
                header_end = True
                break
            if (line[0] == "element") and (line[1] == "vertex"):
                n_vertices = int(line[2])
            if (line[0] == "property") and (line[2] == "label"):
                with_label = True
            if (line[0] == "property") and (line[2] == "nx"):
                with_normal = True
        assert header_end

        if n_vertices == 0:
            print("WARNING: empty ply {}".format(ply_fn))
            if with_label:
                return [], [], []
            else:
                return [], [], None

        if with_normal:
            normals = []
        if with_label:
            labels = []

        # Read vertices
        for line in lines[lineidx+1:lineidx+1+n_vertices]:
            line = line.strip().split(' ')
            assert len(line) >= 3
            vertex = [float(line[0]), float(line[1]), float(line[2])]
            vertices.append(vertex)
            if with_normal:
                normal = [float(line[3]), float(line[4]), float(line[5])]
                normals.append(normal)
            if with_label:
                labels.append(int(line[6]))

    vertices = np.vstack(vertices)
    if with_normal:
        normals = np.vstack(normals)
    else:
        normals = None
    if with_label:
        labels = np.vstack(labels)
    else:
        labels = None

    return vertices, normals, labels


def write_ply_with_normals_and_others(ply_fn, vertices, normals, others, others_names=['label'], others_types=None):
    os.makedirs(os.path.dirname(ply_fn), exist_ok=True)

    assert isinstance(vertices, np.ndarray) and vertices.ndim == 2
    assert isinstance(normals, np.ndarray) and normals.ndim == 2
    assert isinstance(others, np.ndarray) and others.ndim == 2 and len(others_names) == others.shape[1]

    # Create header
    header = 'ply\n' \
             'format ascii 1.0\n' \
             'element vertex ' + str(len(vertices)) + '\n' \
                                                      'property float x\n' \
                                                      'property float y\n' \
                                                      'property float z\n' \
                                                      'property float nx\n' \
                                                      'property float ny\n' \
                                                      'property float nz\n'
    if others_types is None:
        for other_name in others_names:
            header += 'property int {}\n'.format(other_name)
    else:
        for other_name, other_type in zip(others_names, others_types):
            header += 'property {} {}\n'.format(other_type, other_name)
    header += 'end_header\n'

    with open(ply_fn, 'w') as f_ply:
        # Write header
        f_ply.write(header)

        # Write vertices + normals + label
        for idx, vertex in enumerate(vertices):
            row = ' '.join([str(vertex[0]), str(vertex[1]), str(vertex[2]),
                            str(normals[idx, 0]), str(normals[idx, 1]), str(normals[idx, 2])])
            for other in others[idx]:
                row += " " + str(other)
            row += "\n"
            f_ply.write(row)


def read_off(file):
    """
    Reads vertices and faces from an off file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(file)

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

        assert lines[0] == 'OFF'

        parts = lines[1].split(' ')
        assert len(parts) == 3

        num_vertices = int(parts[0])
        assert num_vertices > 0

        num_faces = int(parts[1])
        assert num_faces > 0

        vertices = []
        for i in range(num_vertices):
            vertex = lines[2 + i].split(' ')
            vertex = [float(point) for point in vertex]
            assert len(vertex) == 3

            vertices.append(vertex)

        faces = []
        for i in range(num_faces):
            face = lines[2 + num_vertices + i].split(' ')
            face = [int(index) for index in face]

            assert face[0] == len(face) - 1
            for index in face:
                assert index >= 0 and index < num_vertices

            assert len(face) > 1

            faces.append(face)

        return vertices, faces


def write_full_ply(ply_fn, features, labels, label_names=['label']):
    os.makedirs(os.path.dirname(ply_fn), exist_ok=True)

    assert isinstance(features, np.ndarray) and features.ndim == 2 and features.dtype == float
    assert isinstance(labels, np.ndarray) and labels.ndim == 2 and labels.dtype == int
    assert labels.shape[1] == len(label_names)

    # Create header
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property float nx
property float ny
property float nz
""".format(len(features))

    features_cnt = features.shape[1]

    if features_cnt == 30:
        for pca in range(0, 24):
            header += "property float pca{}\n".format(pca)
    elif features_cnt == 70:
        for curvature in range(0, 64):
            header += "property float curvature{}\n".format(curvature)
    elif features_cnt == 94:
        for curvature in range(0, 64):
            header += "property float curvature{}\n".format(curvature)
        for pca in range(0, 24):
            header += "property float pca{}\n".format(pca)

    for label_name in label_names:
        header += 'property int {}\n'.format(label_name)
    header += 'end_header\n'

    with open(ply_fn, 'w') as f_ply:
        f_ply.write(header)

        for idx in range(len(features)):
            feature = features[idx]
            row = ' '.join(feature.astype(str))
            for label in labels[idx]:
                row += " " + str(label)
            row += "\n"
            f_ply.write(row)


def write_ply(ply_fn, vertices):
    os.makedirs(os.path.dirname(ply_fn), exist_ok=True)

    assert isinstance(vertices, np.ndarray) and vertices.ndim == 2

    # Create header
    header = 'ply\n' \
             'format ascii 1.0\n' \
             'element vertex ' + str(len(vertices)) + '\n' \
                                                      'property float x\n' \
                                                      'property float y\n' \
                                                      'property float z\n' \
                                                      'end_header\n'

    with open(ply_fn, 'w') as f_ply:
        # Write header
        f_ply.write(header)

        # Write vertices
        for idx, vertex in enumerate(vertices):
            row = ' '.join([str(vertex[0]), str(vertex[1]), str(vertex[2])]) + '\n'
            f_ply.write(row)

