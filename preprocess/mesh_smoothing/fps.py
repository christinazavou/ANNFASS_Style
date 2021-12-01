import numpy as np
import torch
from torch_cluster import fps


def farthest_point_sampling(points: np.ndarray, ratio=0.5):
    x = torch.from_numpy(points)
    # batch = torch.tensor([0, 0, 0, 0])
    index = fps(x, ratio=ratio, random_start=False)
    return points[index.numpy()]

