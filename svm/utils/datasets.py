import numpy as np
import pandas as pd
from plyfile import PlyData
from torch.utils.data import DataLoader
import cv2


class StylisticImagesEncodingsDataset(object):

    def __init__(self, csv_file, ignore_in_path=[]):
        self.data_frame = pd.read_csv(csv_file, delimiter=";", header=None)
        if ignore_in_path:
            init_size = self.data_frame.shape[0]
            for ignore_str in ignore_in_path:
                self.data_frame["ignore"] = self.data_frame.apply(lambda x: ignore_str in str(x[1]), axis=1)
                self.data_frame = self.data_frame[self.data_frame["ignore"] == False]
                self.data_frame = self.data_frame.drop('ignore', 1)
            print("ignoring {} rows".format(init_size - self.data_frame.shape[0]))

    def __len__(self):
        return len(self.data_frame) - 1

    def __getitem__(self, idx):
        encoding_path = self.data_frame.iloc[idx + 1, 0]
        labels_path = self.data_frame.iloc[idx + 1, 1]
        encoding = np.load(encoding_path)
        encoding = encoding.reshape((-1))
        labels = np.load(labels_path)
        labels = labels.reshape((-1))
        name = "/".join(encoding_path.split("/")[-2:])[:-4]
        return encoding, labels, name


class StylisticImagesEncodingsDataLoader(DataLoader):

    def collate_fn(self, batch):
        encodings = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        names = [b[2] for b in batch]
        encodings = np.vstack(encodings)
        labels = np.vstack(labels)
        names = np.stack(names)
        if not self.one_hot:
            labels = np.argmax(labels, 1)
        return encodings, labels, names

    def __init__(self, csv_file, batch_size=16, shuffle=True, num_workers=4, one_hot=False, ignore_in_path=[]):
        dataset = StylisticImagesEncodingsDataset(csv_file=csv_file, ignore_in_path=ignore_in_path)
        self.one_hot = one_hot
        super().__init__(dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         collate_fn=self.collate_fn)


class StylisticImagesHogDataset(object):

    def __init__(self, csv_file, ignore_in_path=[]):
        self.data_frame = pd.read_csv(csv_file, delimiter=";", header=None)
        if ignore_in_path:
            init_size = self.data_frame.shape[0]
            for ignore_str in ignore_in_path:
                self.data_frame["ignore"] = self.data_frame.apply(lambda x: ignore_str in str(x[1]), axis=1)
                self.data_frame = self.data_frame[self.data_frame["ignore"] == False]
                self.data_frame = self.data_frame.drop('ignore', 1)
            print("ignoring {} rows".format(init_size - self.data_frame.shape[0]))
        self.hog_descriptor = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 8)

    def __len__(self):
        return len(self.data_frame) - 1

    def __getitem__(self, idx):
        image_path = self.data_frame.iloc[idx + 1, 0]
        labels_path = self.data_frame.iloc[idx + 1, 1]
        image = cv2.imread(image_path)
        resized_img = cv2.resize(image, (128, 128))
        encoding = self.hog_descriptor.compute(resized_img)
        encoding = encoding.reshape((-1))
        labels = np.load(labels_path)
        labels = labels.reshape((-1))
        name = f"{image_path.split('/')[-3]}/{image_path.split('/')[-2].replace('_', '')}_{image_path.split('/')[-1]}"
        return encoding, labels, name


class StylisticImagesHogDataLoader(DataLoader):

    def collate_fn(self, batch):
        encodings = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        names = [b[2] for b in batch]
        encodings = np.vstack(encodings)
        labels = np.vstack(labels)
        names = np.stack(names)
        if not self.one_hot:
            labels = np.argmax(labels, 1)
        return encodings, labels, names

    def __init__(self, csv_file, batch_size=16, shuffle=True, num_workers=4, one_hot=False, ignore_in_path=[]):
        dataset = StylisticImagesHogDataset(csv_file=csv_file, ignore_in_path=ignore_in_path)
        self.one_hot = one_hot
        super().__init__(dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         collate_fn=self.collate_fn)


class StylisticComponentPlyWithCurvaturesDataset(object):

    def __init__(self, csv_file, ignore_in_path=[]):
        self.data_frame = pd.read_csv(csv_file, delimiter=";", header=None)
        if ignore_in_path:
            init_size = self.data_frame.shape[0]
            for ignore_str in ignore_in_path:
                self.data_frame["ignore"] = self.data_frame.apply(lambda x: ignore_str in str(x[1]), axis=1)
                self.data_frame = self.data_frame[self.data_frame["ignore"] == False]
                self.data_frame = self.data_frame.drop('ignore', 1)
            print("ignoring {} rows".format(init_size - self.data_frame.shape[0]))

    def __len__(self):
        return len(self.data_frame) - 1

    def __getitem__(self, idx):
        ply_path = self.data_frame.iloc[idx + 1, 0]
        plydata = PlyData.read(ply_path)
        data = plydata.elements[0].data
        curvature_columns = ['curvature{}'.format(c) for c in range(64)]
        features = []
        for curvature in curvature_columns:
            features.append(data[curvature].astype(np.float))
        features = np.array(features).T
        labels = np.array(data['label'], dtype=np.int32)
        name = ply_path.split("/")[-2] + "/" + ply_path.split("/")[-1][:-4]
        names = [name] * len(labels)
        return features, labels, np.array(names)


class StylisticComponentPlyWithCurvaturesDataLoader(DataLoader):

    def collate_fn(self, batch):
        features = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        names = [b[2] for b in batch]
        features = np.vstack(features)
        labels = np.hstack(labels)
        names = np.hstack(names)
        return features, labels, names

    def __init__(self, csv_file, batch_size=16, shuffle=True, num_workers=4, ignore_in_path=[]):
        self.dataset = StylisticComponentPlyWithCurvaturesDataset(csv_file=csv_file, ignore_in_path=ignore_in_path)
        if batch_size == -1:
            batch_size = len(self.dataset)
        super().__init__(self.dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         collate_fn=self.collate_fn)
