import os
from utils.pytorch3d_vis import CustomSingleViewMeshRenderer
import matplotlib.pyplot as plt
import torch

# data = "furniture"
data = "building_zhaoliang"

preprocessed_data = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/{data}"
ply_data = f"/media/graphicslab/BigData/zavou/ANNFASS_DATA/compressed_files/Data-all/Data/{data}/mesh"

out_dir = f"/media/graphicslab/BigData/zavou/ANNFASS_DATA/compressed_files/Data-all/Data/{data}/renderings"
os.makedirs(out_dir, exist_ok=True)

from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex


def run_scene_element(csvr, mesh_file):
    verts, faces, aux = load_obj(mesh_file,
                                 device=csvr.device,
                                 load_textures=False,
                                 create_texture_atlas=False,)
    verts = verts.to(csvr.device)
    faces_idx = faces.verts_idx.to(csvr.device)

    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    verts_rgb = torch.ones_like(verts)[None]
    textures = TexturesVertex(verts_features=verts_rgb.to(csvr.device))

    meshes = Meshes(verts=[verts], faces=[faces_idx], textures=textures)
    img = csvr(mesh=meshes, show=False)
    return img


def run_scene(scene_element):
    csvr = CustomSingleViewMeshRenderer(torch.device('cuda'), dist=3, elev=30, azim=160)

    mesh_file = os.path.join(preprocessed_data, scene_element, "model.obj")
    img = run_scene_element(csvr, mesh_file)
    out_file = os.path.join(out_dir, f"{scene_element.replace('/', '_')}.png")
    plt.figure()
    plt.imshow(img)
    plt.savefig(out_file)
    plt.close()


with open(f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/{data}_styles32.txt", "r") as fin:
    selected = fin.readlines()

for selected_element in selected:
    selected_element = selected_element.rstrip()
    run_scene(selected_element)
