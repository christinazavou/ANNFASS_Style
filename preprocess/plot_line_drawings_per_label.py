import argparse
import logging
import os
import sys
from os.path import dirname, realpath, join, basename

import cv2
import matplotlib.pyplot as plt

SOURCE_DIR = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(SOURCE_DIR)
from common.utils import set_logger_file, BUILDNET_STYLISTIC_ELEMENTS
from common.matplotlib_utils import image_grid, render_views, figure_to_img


LOGGER = logging.getLogger(__name__)


def get_renderings_from_dir(directory, prefix):
    img0 = cv2.imread(join(directory, f"{prefix}r0.png"))
    img1 = cv2.imread(join(directory, f"{prefix}r1.png"))
    img2 = cv2.imread(join(directory, f"{prefix}r2.png"))
    img3 = cv2.imread(join(directory, f"{prefix}r3.png"))
    img4 = cv2.imread(join(directory, f"{prefix}r4.png"))
    img5 = cv2.imread(join(directory, f"{prefix}.png"))
    return [img0, img1, img2, img3, img4, img5]


def get_component_renderings(component_dir):
    images = get_renderings_from_dir(component_dir, "tr_img")
    if all(img is None for img in images):
        return None, None
    building = basename(dirname(component_dir))
    title = basename(component_dir).replace("style_mesh_", "")
    group = title.split("_")[0]
    title = title.replace(f"{group}_", f"{group}\n")
    style = title.split("_")[-1]
    title = title.replace(f"_{style}", f"\n{style}")
    title = f"{building}\n{title}"
    img = render_views(images, color='white', as_img=True)
    return img, title


def run_semlabel():
    semlabel_images = []
    semlabel_components = []
    for building_dir in os.listdir(components_dir):
        for component_dir in os.listdir(join(components_dir, building_dir)):
            if semantic_label in component_dir:
                img, title = get_component_renderings(join(components_dir, building_dir, component_dir))
                if img is not None:
                    semlabel_images.append(img)
                    semlabel_components.append(title)

    if len(semlabel_images) == 0:
        return

    i = 0
    for idx in range(0, len(semlabel_images), 7):
        i += 1
        img_file = join(out_dir, f"{semantic_label}_components{i}.png")
        current_images = semlabel_images[idx:idx+7]
        current_titles = semlabel_components[idx:idx+7]
        fig = image_grid(current_images, title=semantic_label, titles=current_titles, cols=len(current_images), rows=1)
        img = figure_to_img(fig)
        cv2.imwrite(img_file, img)
    plt.close('all')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-RENDER_MODE', type=int, default=1, help="1: freestyle, 2: materials")
    parser.add_argument('-ROOT_DIR', type=str, required=True)
    parser.add_argument('-COMPONENTS_DIR', type=str, default="unified_normalized_components")
    parser.add_argument('-OUT_DIR', type=str, default="plotted_renderings")
    parser.add_argument('-LOGS_DIR', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.LOGS_DIR):
        os.makedirs(args.LOGS_DIR, exist_ok=True)
    log_file = os.path.join(args.LOGS_DIR, f'{basename(__file__)}.log')
    print("logs in ", log_file)
    set_logger_file(log_file, LOGGER)

    components_dir = join(args.ROOT_DIR, args.COMPONENTS_DIR)
    out_dir = join(args.ROOT_DIR, args.OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    for semantic_label in BUILDNET_STYLISTIC_ELEMENTS:
        run_semlabel()
