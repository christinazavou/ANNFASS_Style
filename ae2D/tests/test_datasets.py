import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ae2D.utils.datasets import StylisticImagesTrainDataset
from ae2D.utils.transformations import ToTensorDict
from svm.utils.datasets import StylisticImagesEncodingsDataLoader


def test_stylistic_image_dataset():
    filepath = 'resources/ANNFASS_materials_on_daylight_all.csv'
    file_data_samples = 34
    transformed_dataset = StylisticImagesTrainDataset(csv_file=filepath,
                                                      transform=transforms.Compose([ToTensorDict()]))
    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

    assert len(dataloader) == np.ceil(file_data_samples / 4)

    for idx, sample_batched in enumerate(dataloader):
        if idx < len(dataloader) - 1:
            assert sample_batched['image'].size() == torch.Size([4, 3, 512, 512])
        else:
            assert sample_batched['image'].size() == torch.Size([file_data_samples % 4, 3, 512, 512])


def test_stylistic_images_encodings_data_loader():
    loader = StylisticImagesEncodingsDataLoader("resources/encodings.csv")
    for batch in loader:
        encodings, labels = batch
        assert encodings.shape[2] == encodings.shape[3]  # width and height are same
        assert len(encodings.shape) == 4  # batch, channels, width, height
        assert len(labels.shape) == 3  # batch, 1, classes
        assert labels.shape[0] == encodings.shape[0]
        break
