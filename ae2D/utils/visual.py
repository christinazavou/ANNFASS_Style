import matplotlib.pyplot as plt
import numpy as np
import torch
import visdom
from torchvision import utils

T_FLOATS = [torch.float, torch.float32, torch.float64]
T_INTS = [torch.int, torch.int32, torch.int64]


def matplotlib_imshow(img, one_channel=False, show=True):
    # NOTE that here you should pass the initial image (i.e. if transforms were applied, the inverse
    # transforms should apply before showing the data
    assert isinstance(img, np.ndarray)
    if one_channel:
        img = np.mean(img, 0)  #img.mean(dim=0)
        plt.imshow(img, cmap="Greys")
    else:
        plt.imshow(img)
    if show:
        plt.show()


def show_batch(sample_batched, show=True):
    images_batch, labels_batch = sample_batched['image'], sample_batched['labels']
    assert torch.is_tensor(images_batch) and torch.is_tensor(labels_batch)

    plt.figure()

    grid = utils.make_grid(images_batch)  # (N, C, H, W)
    plt.imshow(grid.permute(1, 2, 0))

    for i in range(images_batch.shape[0]):
        print(labels_batch[i])
    plt.title('Batch from dataloader')

    plt.axis('off')
    plt.ioff()
    if show:
        plt.show()


def show_initial_and_reconstructed_batch(initial_images, reconstructed_images, epoch, batch, show=True):
    assert torch.is_tensor(initial_images) and torch.is_tensor(reconstructed_images)
    in_values_ok1 = initial_images.dtype in T_FLOATS and initial_images.max() <= 1
    in_values_ok2 = initial_images.dtype in T_INTS and 1 < initial_images.max() <= 255
    assert in_values_ok1 or in_values_ok2
    o_values_ok1 = reconstructed_images.dtype in T_FLOATS and reconstructed_images.max() <= 1
    o_values_ok2 = reconstructed_images.dtype in T_INTS and 1 < reconstructed_images.max() <= 255
    assert o_values_ok1 or o_values_ok2

    grid1 = utils.make_grid(initial_images)  # (N, C, H, W)
    grid2 = utils.make_grid(reconstructed_images)  # (N, C, H, W)
    return grid1, grid2

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.imshow(grid1.permute(1, 2, 0))
    ax2.imshow(grid2.cpu().detach().numpy().transpose(1, 2, 0))
    ax1.axis('off')
    ax2.axis('off')
    plt.ioff()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.title('epoch: {}, batch: {}'.format(epoch, batch))

    if show:
        plt.show()
    else:
        return fig


class ServiceVisualizer(visdom.Visdom):

    def __init__(self, env):
        super().__init__()
        self.env = env

    def show(self, init_images, out_images, epoch, batch):
        if isinstance(init_images, np.ndarray) and isinstance(out_images, np.ndarray):
            assert len(init_images.shape) == 4 and len(out_images.shape) == 4
            init_images = list(init_images)
            out_images = list(out_images)
        else:
            assert isinstance(init_images, list) and isinstance(out_images, list)
            assert len(init_images[0].shape) == 3 and len(out_images[0].shape) == 3

        assert init_images[0].shape[0] in [1, 3] and out_images[0].shape[0] in [1, 3]
        in_values_ok1 = isinstance(init_images[0][0,0,0], np.floating) and np.max(init_images[0]) <= 1
        in_values_ok2 = isinstance(init_images[0][0,0,0], np.integer) and 1 < np.max(init_images[0]) <= 255
        assert in_values_ok1 or in_values_ok2
        o_values_ok1 = isinstance(out_images[0][0,0,0], np.floating) and np.max(out_images[0]) <= 1
        o_values_ok2 = isinstance(out_images[0][0,0,0], np.integer) and 1 < np.max(out_images[0]) <= 255
        assert o_values_ok1 or o_values_ok2

        self.images(init_images + out_images,
                    nrow=len(init_images),
                    win='epoch: {}, batch: {}'.format(epoch, batch),
                    env=self.env,
                    opts={'caption': 'epoch: {}, batch: {}'.format(epoch, batch)})

