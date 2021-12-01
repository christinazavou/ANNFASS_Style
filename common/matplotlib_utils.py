import matplotlib.pyplot as plt
import numpy as np


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
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 10))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    if rows * cols == 1:
        axarr_list = [axarr]
    else:
        axarr_list = axarr.ravel()

    if titles:
        assert len(titles) == len(images)
        for ax, im, subtitle in zip(axarr_list, images, titles):
            if rgb:
                # only render RGB channels
                ax.imshow(im[..., :3])
            else:
                # only render Alpha channel
                ax.imshow(im[..., 3])
            ax.set_title(subtitle, fontsize=8)
            if not show_axes:
                ax.set_axis_off()
    else:
        for ax, im in zip(axarr_list, images):
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
        plt.suptitle(title, color='blue', fontsize=16)

    return fig


def figure_to_img(fig):
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image_from_plot


def render_views(images, title=None, color='b', as_img=True):

    fig = plt.figure(figsize=(3, 3*len(images)))
    gs = fig.add_gridspec(len(images), 1)

    for axi, img in enumerate(images):
        if img is not None:
            ax = fig.add_subplot(gs[axi, 0])
            ax.imshow(img), ax.axis('off')

    if title:
        fig.suptitle(title, color=color, fontsize=16)

    plt.tight_layout()

    if as_img:
        img = figure_to_img(fig)
        plt.close()
        return img
    else:
        return fig


if __name__ == '__main__':
    img1 = np.random.random((100, 100, 3))
    fig = image_grid([img1])
    plt.show()
