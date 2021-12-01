# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image


def image_grid(
    images,
    titles=None,
    title=None,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    if titles:
        assert len(titles) == len(images)
        for ax, im, title in zip(axarr.ravel(), images, titles):
            if rgb:
                # only render RGB channels
                ax.imshow(im[..., :3])
            else:
                # only render Alpha channel
                ax.imshow(im[..., 3])
            ax.set_title(title, fontsize=8)
            if not show_axes:
                ax.set_axis_off()
    else:
        for ax, im in zip(axarr.ravel(), images):
            if rgb:
                # only render RGB channels
                ax.imshow(im[..., :3])
            else:
                # only render Alpha channel
                ax.imshow(im[..., 3])
            if not show_axes:
                ax.set_axis_off()
    plt.tight_layout()
    if title:
        plt.suptitle(title)
    return fig


def render_result(content_images, style_images, generated_images,
                  show=False, save=None, figsize=None,
                  content_titles=('Coarse Content', 'Detailed Content', 'MCubes C-Content', 'MCubes D-Content', 'Content Mesh'),
                  style_titles=('Coarse Style', 'Detailed Style', 'MCubes C-Style', 'MCubes D-Style', 'Style Mesh'),
                  gen_titles=('Coarse Gen', 'Detailed Gen', 'MCubes C-Gen', 'MCubes D-Gen')):

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 5)

    ax_content_0 = fig.add_subplot(gs[0, 0])
    ax_content_0.imshow(content_images[0]), ax_content_0.set_title(content_titles[0]), content_images.axis('off')

    ax_content_1 = fig.add_subplot(gs[0, 1])
    ax_content_1.imshow(content_images[1]), ax_content_1.set_title(content_titles[1]), ax_content_1.axis('off')

    ax_content_2 = fig.add_subplot(gs[0, 2])
    ax_content_2.imshow(content_images[2]), ax_content_2.set_title(content_titles[2]), ax_content_2.axis('off')

    ax_content_3 = fig.add_subplot(gs[0, 3])
    ax_content_3.imshow(content_images[3]), ax_content_3.set_title(content_titles[3]), ax_content_3.axis('off')

    ax_content_4 = fig.add_subplot(gs[0, 4])
    ax_content_4.imshow(content_images[4]), ax_content_4.set_title(content_titles[4]), ax_content_4.axis('off')

    ax_style_0 = fig.add_subplot(gs[1, 0])
    ax_style_0.imshow(style_images[0]), ax_style_0.set_title(style_titles[0]), ax_style_0.axis('off')

    ax_style_1 = fig.add_subplot(gs[1, 1])
    ax_style_1.imshow(style_images[1]), ax_style_1.set_title(style_titles[1]), ax_style_1.axis('off')

    ax_style_2 = fig.add_subplot(gs[1, 2])
    ax_style_2.imshow(style_images[2]), ax_style_2.set_title(style_titles[2]), ax_style_2.axis('off')

    ax_style_3 = fig.add_subplot(gs[1, 3])
    ax_style_3.imshow(style_images[3]), ax_style_3.set_title(style_titles[3]), ax_style_3.axis('off')

    ax_style_4 = fig.add_subplot(gs[1, 4])
    ax_style_4.imshow(style_images[3]), ax_style_4.set_title(style_titles[4]), ax_style_4.axis('off')

    ax_gen_0 = fig.add_subplot(gs[2, 0])
    ax_gen_0.imshow(generated_images[0]), ax_gen_0.set_title(gen_titles[0]), ax_gen_0.axis('off')

    ax_gen_1 = fig.add_subplot(gs[1, 1])
    ax_gen_1.imshow(generated_images[1]), ax_gen_1.set_title(gen_titles[1]), ax_gen_1.axis('off')

    ax_gen_2 = fig.add_subplot(gs[1, 3])
    ax_gen_2.imshow(generated_images[2]), ax_gen_2.set_title(gen_titles[2]), ax_gen_2.axis('off')

    ax_gen_3 = fig.add_subplot(gs[1, 3])
    ax_gen_3.imshow(generated_images[3]), ax_gen_3.set_title(gen_titles[3]), ax_gen_3.axis('off')

    plt.tight_layout()
    if show:
        plt.show()
    if save:
        assert save.endswith(".png")
        plt.savefig(save)
        plt.close()


def fig_to_img(fig, img_size=None):
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if img_size:
        image_from_plot = Image.fromarray(image_from_plot)
        image_from_plot = np.asarray(image_from_plot.resize((img_size, img_size)).convert('L'))
    return image_from_plot


def render_views(images, show=False, save=None, title=None, as_img=True, img_size=None):

    fig = plt.figure(figsize=(3, 3*len(images)))
    gs = fig.add_gridspec(len(images), 1)

    for axi, img in enumerate(images):
        if img is not None:
            ax = fig.add_subplot(gs[axi, 0])
            ax.imshow(img), ax.axis('off')

    if title:
        fig.suptitle(title)

    plt.tight_layout(), plt.axis('off')

    if show:
        plt.show()
    if save:
        assert save.endswith(".png")
        plt.savefig(save)

    if as_img:
        img = fig_to_img(fig, img_size)
        plt.close()
        return img

    return fig


def render_example(images,
                   show=False, save=None, title=None, figsize=None, as_img=True,
                   titles=('Coarse', 'Detailed', 'MCubes Coarse', 'MCubes Detailed', 'Input Mesh', 'Reference Mesh')):

    if figsize is None:
        figsize = (len(images)*3*len(images), len(images)*3)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, len(images))

    for axi, (img, title) in enumerate(zip(images, titles)):
        if img is not None:
            ax = fig.add_subplot(gs[0, axi])
            ax.imshow(img), ax.set_title(title), ax.axis('off')

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if show:
        plt.show()
    if save:
        assert save.endswith(".png")
        plt.savefig(save)
    if as_img:
        img = fig_to_img(fig)
        plt.close()
        return img
    return fig


def plot_matrix(mat, show=False, title=None, as_img=False):
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    pos = ax.imshow(mat, cmap='hot', interpolation='nearest')
    fig.colorbar(pos, ax=ax)
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    if as_img:
        img = fig_to_img(fig)
        return img
    return fig


if __name__ == '__main__':
    import cv2
    m = np.random.random((10, 8))
    # im = plot_matrix(m, show=True, title='latent')
    # im = plot_matrix(m, )
    # plt.show()
    im = plot_matrix(m, title='latent', as_img=True)
    cv2.imwrite("latent.png", im)

