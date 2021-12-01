import os
import sys

import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from ae2D.utils.transformations import ToTensorDict, IntervalMappingDict, interval_mapping

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from common.utils import STYLES as classes


class StylisticImagesTrainDataset(Dataset):

    def __init__(self, root_dir, txt_file, transform=None):
        self.root_dir = root_dir
        with open(txt_file, "r") as fin:
            files = [line.rstrip() for line in fin.readlines()]
            files = [os.path.join(self.root_dir, f) for f in files]
            self.files = [f for f in files if os.path.exists(f)]
        print("loaded {} files".format(len(self.files)))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        img_name = self.files[idx]
        image = io.imread(img_name)
        sample = {'image': image}
        if self.transform:
            sample = self.transform(sample)
        return sample


class StylisticImagesTrainDataLoader(DataLoader):
    def __init__(self, root_dir, txt_file, batch_size=16, num_workers=4):
        transform = transforms.Compose([
            ToTensorDict(has_label=False),
            IntervalMappingDict(0, 255, 0, 1)
            # NormalizeDict((169.85, 169.33,  164.46), (23.11, 24.28, 27.49)),
            # ToOneChannelDict()
        ])
        dataset = StylisticImagesTrainDataset(root_dir, txt_file, transform)
        print("dataset size: {}".format(len(dataset)))
        super().__init__(dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=num_workers)


class StylisticImagesTestDataset(Dataset):

    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file, header=None, delimiter=";")
        print("loaded data frame shape {}".format(self.data_frame.shape))
        self.root_dir = self.data_frame.iloc[0, 0]
        names = list(self.data_frame.iloc[1:][1].unique())
        keys = [None]
        for idx, row in self.data_frame.iterrows():
            if idx == 0:
                continue
            keys.append(names.index(row[1]))
        self.data_frame['key'] = keys

    def __len__(self):
        return len(self.data_frame[1].unique()) - 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        def one_hot(class_idx):
            v = np.zeros((1, len(classes)))
            v[0, class_idx] = 1
            return v

        rows = self.data_frame[self.data_frame['key'] == idx]
        img_names = list(rows[0])
        keys = list(rows[1])
        assert len(set(keys)) == 1
        key = keys[0]
        images = [[io.imread(os.path.join(self.root_dir, img_name))] for img_name in img_names]
        label = int(list(rows[2])[0])
        labels = [[one_hot(label)] for _ in img_names]
        images = np.concatenate(images)
        labels = np.concatenate(labels)

        sample = {'image': images, 'labels': labels, 'key': key}

        return sample


class StylisticImagesTestDataLoader(DataLoader):

    from_min = 0.
    from_max = 255.
    to_min = 0.
    to_max = 1.

    @staticmethod
    def collate_fn(batch):
        assert len(batch) == 1
        images = batch[0]['image']
        labels = batch[0]['labels']
        key = batch[0]['key']
        preprocess_image = lambda x: torch.from_numpy(
                                        interval_mapping(
                                            x.transpose((2, 0, 1)),
                                            StylisticImagesTestDataLoader.from_min,
                                            StylisticImagesTestDataLoader.from_max,
                                            StylisticImagesTestDataLoader.to_min,
                                            StylisticImagesTestDataLoader.to_max))
        images = [preprocess_image(im) for im in images]
        return {'image': torch.stack(images, 0), 'labels': torch.from_numpy(labels), 'key': key}

    def __init__(self, csv_file, num_workers=4):
        dataset = StylisticImagesTestDataset(csv_file=csv_file)
        print("dataset size (unique elements): {}".format(len(dataset)))
        super().__init__(dataset,
                         batch_size=1,
                         shuffle=False,
                         num_workers=num_workers,
                         collate_fn=StylisticImagesTestDataLoader.collate_fn)
