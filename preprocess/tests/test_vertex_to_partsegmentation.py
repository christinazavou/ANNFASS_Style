from preprocess.buildnet.vertex_to_partsegmentation import process_one
from preprocess.minkowski_encoding.utils import read_obj, read_obj_with_components
import numpy as np


def test_process_one():
    for filename in ['RESIDENTIALvilla_mesh6005_label.json', 'RESIDENTIALvilla_mesh6029_label.json',
                     'RESIDENTIALvilla_mesh6047_label.json', 'RESIDENTIALvilla_mesh6062_label.json']:
        process_one("resources/textures", "resources/component_to_labels", "resources/pycollada_unit_obj",
                    "resources/normalizedObj", filename)
        obj_name = filename.replace("_label.json", ".obj")
        obj_before = read_obj("resources/pycollada_unit_obj/{}".format(obj_name))
        obj_after = read_obj("resources/normalizedObj/{}/{}".format(obj_name.replace(".obj", ""), obj_name))
        assert np.array_equal(obj_before[0], obj_after[0])
        assert np.array_equal(obj_before[1], obj_after[1])
        assert np.array_equal(obj_before[2], obj_after[2])


def test_read_obj_with_components():
    filename = 'RESIDENTIALvilla_mesh6005_label.json'
    process_one("resources/textures", "resources/component_to_labels", "resources/pycollada_unit_obj",
                "resources/normalizedObj", filename)
    obj_name = filename.replace("_label.json", ".obj")
    obj_after = read_obj_with_components("resources/normalizedObj/{}/{}".format(obj_name.replace(".obj", ""), obj_name))
    assert np.max(obj_after[2]) <= obj_after[3].shape[0]
