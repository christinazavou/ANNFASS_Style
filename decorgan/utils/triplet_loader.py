import os

import numpy as np
import torch.utils.data


class BaseLoader(torch.utils.data.Dataset):
    def __init__(self, triplet_path, preprocessed_path,):
        self.triplet_path = triplet_path
        self.preprocessed_path = preprocessed_path
        self.triplet_paths = []
        paths = {}
        for d in os.listdir(self.preprocessed_path):
            binvox_file = os.path.join(self.preprocessed_path, d, "whole/model_filled.binvox")
            if os.path.exists(binvox_file):
                try:
                    shape_scene, shape_exact = d.split("-")
                    paths[f"{shape_scene}/{d}"] = binvox_file
                except:
                    paths[d] = binvox_file
        with open(triplet_path, "r") as fin:
            triplet_lines = fin.readlines()
        for triplet in triplet_lines:
            triplet = triplet.rstrip().split(" ")
            try:
                triplet = [paths[triplet[0]], paths[triplet[1]], paths[triplet[2]]]
                self.triplet_paths.append(triplet)
            except:
                print("some path not in encodings ..")
                pass

    def __getitem__(self, index):
        anchor_path, pos_path, neg_path = self.triplet_paths[index]
        return anchor_path, pos_path, neg_path

    def __len__(self):
        return len(self.triplet_paths)


def get_loader(preprocessed_path, triplet_path, batch_size=64):
    kwargs = {'num_workers': 0, 'pin_memory': True}

    data_loader = torch.utils.data.DataLoader(
        BaseLoader(triplet_path, preprocessed_path),
        batch_size=batch_size, shuffle=True, **kwargs)

    return data_loader
