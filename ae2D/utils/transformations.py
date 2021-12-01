import os

import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader


class DataSetUtils:
    @staticmethod
    def get_mean_function(data_loader, function=np.mean, from_dict=False):
        batch_mean_function_image_per_channel = []
        for batch in data_loader:
            if from_dict:
                images, labels = batch['image'], batch['labels']
            else:
                images, labels = batch
            batch_view = images.view(images.size(0), images.size(1), -1)
            image_function_per_channel = function(batch_view.numpy(), axis=2)
            image_mean_function_per_channel = np.mean(image_function_per_channel, 0)
            batch_mean_function_image_per_channel.append(image_mean_function_per_channel)
        batch_mean_function_image_per_channel = np.array(batch_mean_function_image_per_channel)
        return np.mean(batch_mean_function_image_per_channel, 0)


class DatasetFromNumpy(Dataset):

    def __init__(self, numpy_data_x, numpy_data_y, transform=None):
        self.transform = transform
        self.x = numpy_data_x
        self.y = numpy_data_y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        image = self.x[idx]
        labels = self.y[idx]
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


class DataSetHelper:

    @staticmethod
    def get_dataloader_from_examples(numpy_data_x, numpy_data_y):

        tensor_x = torch.Tensor(numpy_data_x)
        tensor_y = torch.Tensor(numpy_data_y)

        t_dataset = TensorDataset(tensor_x, tensor_y)
        t_dataloader = DataLoader(t_dataset)

        return t_dataloader


class InverseNormalizeDict(object):

    def __init__(self, mean, std):
        assert isinstance(mean, (np.ndarray, tuple, list)) and len(mean) == 1 or len(mean) == 3
        assert isinstance(std, (np.ndarray, tuple, list)) and len(mean) == 1 or len(mean) == 3
        self.mean = np.array(mean)[:, np.newaxis, np.newaxis]
        self.std = np.array(std)[:, np.newaxis, np.newaxis]
        if (self.std == 0).any():
            raise ValueError('std evaluated to zero, leading to division by zero.')

    def __call__(self, sample):
        assert isinstance(sample, dict) and 'image' in sample and 'labels' in sample
        image, labels = sample['image'], sample['labels']
        assert isinstance(image, np.ndarray) and isinstance(labels, np.ndarray), "Not accepted input type."
        assert len(image.shape) == 3 and (image.shape[0] == 1 or image.shape[0] == 3)
        image = image * self.std + self.mean
        return {
            'image': image,
            'labels': labels
        }


class NormalizeDict(object):

    def __init__(self, mean, std):
        assert isinstance(mean, (np.ndarray, tuple, list)) and len(mean) == 1 or len(mean) == 3
        assert isinstance(std, (np.ndarray, tuple, list)) and len(mean) == 1 or len(mean) == 3
        self.mean = np.array(mean)[:, np.newaxis, np.newaxis]
        self.std = np.array(std)[:, np.newaxis, np.newaxis]
        if (self.std == 0).any():
            raise ValueError('std evaluated to zero, leading to division by zero.')

    def __call__(self, sample):
        assert isinstance(sample, dict) and 'image' in sample and 'labels' in sample
        image, labels = sample['image'], sample['labels']
        if isinstance(image, np.ndarray) and isinstance(labels, np.ndarray):
            assert len(image.shape) == 3 and image.shape[0] in [1, 3]
        else:
            assert len(image.size()) == 3 and image.size(0) in [1, 3]
        image = (image - self.mean) / self.std
        return {
            'image': image,
            'labels': labels
        }


class InverseToTensorDict(object):

    def __call__(self, samples):
        images, labels = samples['image'], samples['labels']

        # swap color axis because: numpy image is H x W x C and torch image is C X H X W
        images = images.numpy().transpose((0, 2, 3, 1))
        return {
            'image': images,
            'labels': labels.numpy()
        }


class ToTensorDict(object):

    def __init__(self, from_dict=True, keep_shape=False, has_label=True):
        self.from_dict = from_dict
        self.keep_shape = keep_shape
        self.has_label = has_label

    def __call__(self, sample):
        if self.from_dict:
            if self.has_label:
                image, labels = sample['image'], sample['labels']
            else:
                image = sample['image']
        else:
            if self.has_label:
                image, labels = sample
            else:
                image = sample

        if not self.keep_shape:
            # swap color axis because: numpy image is H x W x C and torch image is C X H X W
            image = image.transpose((2, 0, 1))

        if self.has_label:
            return {
                'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)
            }
        else:
            return {
                'image': torch.from_numpy(image),
                'labels': torch.from_numpy(np.ones((len(image))) * -1)
            }


class ToOneChannelDict(object):

    def __call__(self, sample):
        assert isinstance(sample, dict)
        image, labels = sample['image'], sample['labels']
        assert torch.is_tensor(image) and torch.is_tensor(labels)
        the_type = image.dtype
        if the_type in [torch.uint8, torch.int32, torch.int64]:
            image = torch.mean(image.type(torch.FloatTensor), 0, True).type(the_type)
        else:
            image = torch.mean(image, 0, True)

        return {
            'image': image,
            'labels': labels
        }


class IntervalMappingDict(object):

    def __init__(self, from_min, from_max, to_min, to_max):
        self.from_min = float(from_min)
        self.to_min = float(to_min)
        self.from_range = float(from_max - from_min)
        self.to_range = float(to_max - to_min)

    def __call__(self, sample):
        assert isinstance(sample, dict)
        image, labels = sample['image'], sample['labels']
        assert torch.is_tensor(image) and torch.is_tensor(labels)
        scaled = torch.div(torch.sub(image, self.from_min), self.from_range)
        image = self.to_min + (scaled * self.to_range)
        return {
            'image': image,
            'labels': labels
        }


def interval_mapping(image, from_min, from_max, to_min, to_max):
    assert isinstance(image, np.ndarray)
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)


def normalize(image, mean, std):
    return (image - mean) / std


class StylisticImagesDatasetFromCsv(Dataset):

    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file, header=None)
        self.root_dir = self.data_frame.iloc[0, 0]
        self.transform = transform

    def __len__(self):
        return len(self.data_frame) - 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx + 1, 0])
        image = io.imread(img_name)
        labels = np.zeros((1,5))
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

