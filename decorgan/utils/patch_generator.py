import torch
import random
import numpy as np
import itertools


def get_random_non_cube_patches(vox: torch.FloatTensor, patch_factor=2, patch_num=8, stride_factor=2):
    """
    Returns patches with random order
    input must be of shape [dim, x,y,z]
    and will return [num_patches, dim, patch_size_x, patch_size_y, patch_size_z]
    """
    patch_shape = np.array(vox[0].shape) // patch_factor
    patch_strides = patch_shape // stride_factor
    expected = np.floor(np.divide(np.array(vox[0].shape) - np.array(patch_shape), np.array(patch_strides)) + 1)
    expected = expected[0] * expected[1] * expected[2]

    patches = vox.unfold(1, patch_shape[0], patch_strides[0])\
                 .unfold(2, patch_shape[1], patch_strides[1])\
                 .unfold(3, patch_shape[2], patch_strides[2])
    patches = patches.contiguous().view(-1, vox.shape[0], patch_shape[0], patch_shape[1], patch_shape[2])
    assert patches.shape[0] == expected
    indices = list(range(patches.shape[0]))
    random.shuffle(indices)
    selected_indices = indices[:patch_num]
    patches = patches[selected_indices]
    return patches


# def get_8_patches(vox, pool):
#     _, _, dim_o_x, dim_o_y, dim_o_z = vox.shape
#     patch1 = vox[0, :-1, :dim_o_x // 2, :dim_o_y // 2, :dim_o_z // 2]
#     patch2 = vox[0, :-1, dim_o_x // 2:, :dim_o_y // 2, :dim_o_z // 2]
#     patch3 = vox[0, :-1, :dim_o_x // 2, dim_o_y // 2:, :dim_o_z // 2]
#     patch4 = vox[0, :-1, :dim_o_x // 2, :dim_o_y // 2, dim_o_z // 2:]
#     patch5 = vox[0, :-1, dim_o_x // 2:, dim_o_y // 2:, :dim_o_z // 2]
#     patch6 = vox[0, :-1, :dim_o_x // 2, dim_o_y // 2:, dim_o_z // 2:]
#     patch7 = vox[0, :-1, dim_o_x // 2:, :dim_o_y // 2, dim_o_z // 2:]
#     patch8 = vox[0, :-1, dim_o_x // 2:, dim_o_y // 2:, dim_o_z // 2:]
#     patch1 = pool(patch1).squeeze(4).squeeze(3).squeeze(2)
#     patch2 = pool(patch2).squeeze(4).squeeze(3).squeeze(2)
#     patch3 = pool(patch3).squeeze(4).squeeze(3).squeeze(2)
#     patch4 = pool(patch4).squeeze(4).squeeze(3).squeeze(2)
#     patch5 = pool(patch5).squeeze(4).squeeze(3).squeeze(2)
#     patch6 = pool(patch6).squeeze(4).squeeze(3).squeeze(2)
#     patch7 = pool(patch7).squeeze(4).squeeze(3).squeeze(2)
#     patch8 = pool(patch8).squeeze(4).squeeze(3).squeeze(2)
#     return torch.cat([patch1, patch2, patch3, patch4, patch5, patch6, patch7, patch8])


def get_random_non_cube_triplet_patches(vox1: torch.FloatTensor, vox2: torch.FloatTensor,
                                        patch_factor=2, stride_factor=2, num_triplets=128):
    """
    Returns patches with random order
    input must be of shape [dim, x,y,z]
    and will return [num_patches, dim, patch_size_x, patch_size_y, patch_size_z]
    """
    assert vox1.shape[0] == vox2.shape[0] == 1 and vox1.ndim == vox2.ndim == 5

    patch_shape = np.array(vox1[0, 0].shape) // patch_factor
    patch_strides = patch_shape // stride_factor
    expected = np.floor(np.divide(np.array(vox1[0, 0].shape) - np.array(patch_shape), np.array(patch_strides)) + 1)
    expected = expected[0] * expected[1] * expected[2]

    patches = vox1.unfold(2, patch_shape[0], patch_strides[0])\
                 .unfold(3, patch_shape[1], patch_strides[1])\
                 .unfold(4, patch_shape[2], patch_strides[2])
    patches1 = patches.contiguous().view(-1, vox1.shape[1], patch_shape[0], patch_shape[1], patch_shape[2])
    assert patches1.shape[0] == expected

    patch_shape = np.array(vox2[0, 0].shape) // patch_factor
    patch_strides = patch_shape // stride_factor
    expected = np.floor(np.divide(np.array(vox2[0, 0].shape) - np.array(patch_shape), np.array(patch_strides)) + 1)
    expected = expected[0] * expected[1] * expected[2]

    patches = vox2.unfold(2, patch_shape[0], patch_strides[0])\
                 .unfold(3, patch_shape[1], patch_strides[1])\
                 .unfold(4, patch_shape[2], patch_strides[2])
    patches2 = patches.contiguous().view(-1, vox2.shape[1], patch_shape[0], patch_shape[1], patch_shape[2])
    assert patches2.shape[0] == expected

    indices1 = list(range(patches1.shape[0]))
    indices2 = list(range(patches2.shape[0]))

    pairs_positives = list(itertools.combinations(indices1, r=2))
    triplets = list(itertools.product(pairs_positives, indices2))
    triplets = [[i, j, k] for ((i, j), k) in triplets]

    random.shuffle(triplets)

    selected_triplets = triplets[:num_triplets]
    anchors = [i for [i,j,k] in selected_triplets]
    positives = [j for [i,j,k] in selected_triplets]
    negatives = [k for [i,j,k] in selected_triplets]

    anchors = patches1[anchors]
    positives = patches1[positives]
    negatives = patches2[negatives]

    return anchors, positives, negatives


def get_random_non_cube_pair_patches(vox1: torch.FloatTensor, vox2: torch.FloatTensor,
                                     patch_factor=2, stride_factor=2, num_pairs=32):
    """
    Returns patches with random order
    input must be of shape [dim, x,y,z]
    and will return [num_patches, dim, patch_size_x, patch_size_y, patch_size_z]
    """
    assert vox1.shape[0] == vox2.shape[0] == 1 and vox1.ndim == vox2.ndim == 5

    patch_shape = np.array(vox1[0, 0].shape) // patch_factor
    patch_strides = patch_shape // stride_factor
    expected = np.floor(np.divide(np.array(vox1[0, 0].shape) - np.array(patch_shape), np.array(patch_strides)) + 1)
    expected = expected[0] * expected[1] * expected[2]

    patches = vox1.unfold(2, patch_shape[0], patch_strides[0])\
                 .unfold(3, patch_shape[1], patch_strides[1])\
                 .unfold(4, patch_shape[2], patch_strides[2])
    patches1 = patches.contiguous().view(-1, vox1.shape[1], patch_shape[0], patch_shape[1], patch_shape[2])
    assert patches1.shape[0] == expected

    patch_shape = np.array(vox2[0,0].shape) // patch_factor
    patch_strides = patch_shape // stride_factor
    expected = np.floor(np.divide(np.array(vox2[0,0].shape) - np.array(patch_shape), np.array(patch_strides)) + 1)
    expected = expected[0] * expected[1] * expected[2]

    patches = vox2.unfold(2, patch_shape[0], patch_strides[0])\
                 .unfold(3, patch_shape[1], patch_strides[1])\
                 .unfold(4, patch_shape[2], patch_strides[2])
    patches2 = patches.contiguous().view(-1, vox2.shape[1], patch_shape[0], patch_shape[1], patch_shape[2])
    assert patches2.shape[0] == expected

    indices1 = list(range(patches1.shape[0]))
    indices2 = list(range(patches2.shape[0]))

    pairs = list(itertools.product(indices1, indices2))

    random.shuffle(pairs)

    selected_pairs = pairs[:num_pairs]
    indices1 = [i for [i,j] in selected_pairs]
    indices2 = [j for [i,j] in selected_pairs]

    patches1 = patches1[indices1]
    patches2 = patches2[indices2]

    return patches1, patches2
