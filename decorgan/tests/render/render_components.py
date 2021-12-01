import os
from utils.pytorch3d_vis import CustomSingleViewMeshRenderer, CustomDefinedViewMeshRenderer
import numpy as np
from utils.matplotlib_utils import image_grid
import matplotlib.pyplot as plt
import cv2
import torch

data = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/data/buildnet_component_v2_orient"
preprocessed_data = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/buildnet_component_refined_v2"
objs = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/normalizedObj_refinedTextures_Local"

out_dir = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/v2_orient_colour_renderings"
os.makedirs(out_dir, exist_ok=True)

from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex, TexturesUV, TexturesAtlas


def run_component(csvr, mesh_file):
    verts, faces, aux = load_obj(mesh_file,
                                 device=csvr.device,
                                 load_textures=True,
                                 create_texture_atlas=True,)
    tex = TexturesAtlas([aux.texture_atlas.to(csvr.device)])

    verts = verts.to(csvr.device)
    faces_idx = faces.verts_idx.to(csvr.device)

    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    meshes = Meshes(verts=[verts], faces=[faces_idx], textures=tex)
    img = csvr(mesh=meshes, show=False)
    return img


def run_building(building):
    csvr = CustomSingleViewMeshRenderer(torch.device('cuda:0'), elev=30, azim=160)
    cdvr = CustomDefinedViewMeshRenderer(6, False)
    building_imgs = []
    building_components = []

    if not os.path.exists(os.path.join(objs, building, f"{building}.obj")):
        return
    if not os.path.exists(os.path.join(preprocessed_data, building)):
        return
    print(building)

    if not os.path.exists(os.path.join(out_dir, f"{building}_mv.png")):
        try:
            img = cdvr(obj_filename=os.path.join(objs, building, f"{building}.obj"), show=False)
            cv2.imwrite(os.path.join(out_dir, f"{building}_mv.png"), img)
        except:
            pass
    if not os.path.exists(os.path.join(out_dir, f"{building_dir}.png")):
        for component_dir in os.listdir(os.path.join(data, building)):
            if component_dir in os.listdir(os.path.join(preprocessed_data, building)):
                mesh_file = os.path.join(data, building, component_dir, "model.obj")
                name = component_dir.split("style_mesh_")[1].replace("group", "g")
                img = run_component(csvr, mesh_file)
                building_imgs.append(img)
                building_components.append(name)
        if len(building_imgs) > 25:
            print(f"Should make more images for building {building_dir}")
            building_imgs = building_imgs[:25]
            building_components = building_components[:25]
            row = int(np.ceil(len(building_imgs) / 5))
            image_grid(building_imgs, building_components, rows=row, cols=5)
            out_file = os.path.join(out_dir, f"{building_dir}.png")
            plt.savefig(os.path.join(out_dir, out_file))
        else:
            building_imgs = building_imgs[:25]
            building_components = building_components[:25]
            row = int(np.ceil(len(building_imgs) / 5))
            image_grid(building_imgs, building_components, rows=row, cols=5)
            out_file = os.path.join(out_dir, f"{building_dir}.png")
            plt.savefig(os.path.join(out_dir, out_file))


for building_dir in os.listdir(data):
    run_building(building_dir)
