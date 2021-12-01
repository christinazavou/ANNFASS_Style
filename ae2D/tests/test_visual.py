import numpy as np
import torch
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader

from ae2D.utils.datasets import StylisticImagesTrainDataset, StylisticImagesTestDataLoader
from ae2D.utils.transformations import DatasetFromNumpy, ToTensorDict, InverseToTensorDict
from ae2D.utils.visual import matplotlib_imshow, show_batch, ServiceVisualizer

img1 = np.array([
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], ],
    [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], ],
    [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], ]
]).transpose((1, 2, 0))
img2 = np.array([
    [[105, 105, 105, 105], [105, 105, 105, 105], [105, 105, 105, 105], [105, 105, 105, 105], ],
    [[200, 200, 200, 200], [200, 200, 200, 200], [200, 200, 200, 200], [200, 200, 200, 200], ],
    [[40, 40, 40, 40], [40, 40, 40, 40], [40, 40, 40, 40], [40, 40, 40, 40], ]
]).transpose((1, 2, 0))

test_images = np.array([img1, img2])
print("test_images ", test_images.shape)
test_labels = np.array([np.array([4.]), np.array([2.])])
print("test_labels ", test_labels.shape)


def test_matplotlib_imshow():
    transform = transforms.Compose([
        ToTensorDict(),
    ])
    inverse_transforms = transforms.Compose([InverseToTensorDict()])
    test_data = DatasetFromNumpy(test_images, test_labels, transform)
    test_loader = DataLoader(test_data, batch_size=2, num_workers=0)
    for data in test_loader:
        data = inverse_transforms(data)
        images = data['image']
        labels = data['labels']
        assert images.shape == (2, 4, 4, 3)
        assert isinstance(images, np.ndarray) and isinstance(labels, np.ndarray)
        matplotlib_imshow(images[0])
        break


def test_show_batch():
    transform = transforms.Compose([
        ToTensorDict(),
    ])
    test_data = DatasetFromNumpy(test_images, test_labels, transform)
    test_loader = DataLoader(test_data, batch_size=2, num_workers=0)

    for sample_batched in test_loader:
        show_batch(sample_batched)
        break
    transformed_dataset = StylisticImagesTrainDataset(csv_file='resources/ANNFASS_val.csv',
                                                      transform=transform)
    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

    for idx, sample_batched in enumerate(dataloader):
        if idx == 3:
            show_batch(sample_batched)


def test_show_test_batch():
    dataloader = StylisticImagesTestDataLoader(csv_file='resources/ANNFASS_val.csv', num_workers=4)
    assert len(dataloader) == 126
    for sample_batched in dataloader:
        assert sample_batched['image'].size() == torch.Size([7, 3, 512, 512])
        assert sample_batched['labels'].size() == torch.Size([7, 1, 5])
        show_batch(sample_batched)
        break


def test_ServiceVisualizer():
    img = io.imread("../resources/rendered_component_example.jpg")
    init_images = np.concatenate([[img], [img]]).transpose((0, 3, 1, 2))
    init_images = list(init_images)
    print("init_images: ", len(init_images), init_images[0].shape)
    vis = ServiceVisualizer("dokimi")
    vis.show(init_images, init_images, 1, 1)

    img = io.imread("../resources/rendered_component_example.jpg")
    img = img / 255.
    init_images = np.concatenate([[img], [img]]).transpose((0, 3, 1, 2))
    vis = ServiceVisualizer("dokimi")
    vis.show(list(init_images), list(init_images), 2, 1)

    img = io.imread("../resources/rendered_component_example.jpg")
    init_images = np.concatenate([[img], [img]]).transpose((0, 3, 1, 2))
    print("init_images: ", init_images.shape)
    vis = ServiceVisualizer("dokimi")
    vis.show(init_images, init_images, 3, 1)
