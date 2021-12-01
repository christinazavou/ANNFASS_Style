import os

import numpy as np
import torch.utils.data


class BaseLoader(torch.utils.data.Dataset):
    def __init__(self, triplet_path, encodings_path, transform=None):
        self.triplet_path = triplet_path
        self.encodings_path = encodings_path
        self.pair_paths_with_label = []
        paths = {}
        for d in os.listdir(self.encodings_path):
            encoding_file = os.path.join(self.encodings_path, d, "whole.npy")
            if os.path.exists(encoding_file):
                try:
                    shape_scene, shape_exact = d.split("-")
                    paths[f"{shape_scene}/{d}"] = encoding_file
                except:
                    paths[d] = encoding_file
        with open(triplet_path, "r") as fin:
            triplet_lines = fin.readlines()
        for triplet in triplet_lines:
            triplet = triplet.rstrip().split(" ")
            try:
                self.pair_paths_with_label.append([paths[triplet[0]], paths[triplet[1]], 1])
                self.pair_paths_with_label.append([paths[triplet[0]], paths[triplet[2]], -1])
            except:
                print("some path not in encodings ..")
                pass

    def __getitem__(self, index):
        anchor_path, query_path, query_label = self.pair_paths_with_label[index]
        anchor = np.load(anchor_path).squeeze()
        query = np.load(query_path).squeeze()
        return anchor, query, query_label

    def __len__(self):
        return len(self.pair_paths_with_label)


def get_loader(args, triplet_train_path, triplet_test_path):
    kwargs = {'num_workers': 0, 'pin_memory': True}

    train_data_loader = torch.utils.data.DataLoader(
        BaseLoader(triplet_train_path, args.encodings_path),
        batch_size=50000, shuffle=True, **kwargs)
    test_data_loader = torch.utils.data.DataLoader(
        BaseLoader(triplet_test_path, args.encodings_path),
        batch_size=50000, shuffle=False, **kwargs)

    return train_data_loader, test_data_loader
