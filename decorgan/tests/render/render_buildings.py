import os
from utils.pytorch3d_vis import CustomSingleViewMeshRenderer, CustomDefinedViewMeshRenderer
import numpy as np
from utils.matplotlib_utils import image_grid
import matplotlib.pyplot as plt
import cv2
import torch

preprocessed_data = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/buildnet_buildings/normalizedObj"

out_dir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/buildnet_buildings/visual"
os.makedirs(out_dir, exist_ok=True)


def run_building(building):
    cdvr = CustomDefinedViewMeshRenderer(torch.device('cuda'), 6, False)

    if not os.path.exists(os.path.join(preprocessed_data, building)):
        return
    print(building)

    if not os.path.exists(os.path.join(out_dir, f"{building}_mv.png")):
        try:
            img = cdvr(obj_filename=os.path.join(preprocessed_data, building, f"model.obj"), show=False)
            cv2.imwrite(os.path.join(out_dir, f"{building}_mv.png"), img)
        except:
            pass


for building_dir in os.listdir(preprocessed_data):
    run_building(building_dir)
