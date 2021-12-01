from unittest import TestCase
from utils.patch_generator import get_random_non_cube_patches, get_random_non_cube_triplet_patches, \
    get_random_non_cube_pair_patches
import torch


class Test(TestCase):
    def test_get_random_non_cube_patches(self):
        voxel = torch.rand((1, 4, 100, 60, 80))
        patches = get_random_non_cube_patches(voxel[0, 0:-1], patch_factor=2, patch_num=8, stride_factor=1)
        print(f"D_out patches: {patches.shape}")
        voxel = torch.rand((1, 4, 50, 60, 40))
        patches = get_random_non_cube_patches(voxel[0, 0:-1], patch_factor=2, patch_num=8, stride_factor=1)
        print(f"D_out patches: {patches.shape}")
        voxel = torch.rand((1, 4, 10, 20, 10))
        patches = get_random_non_cube_patches(voxel[0, 0:-1], patch_factor=2, patch_num=8, stride_factor=1)
        print(f"D_out patches: {patches.shape}")
        voxel = torch.rand((1, 4, 32, 32, 32))
        patches = get_random_non_cube_patches(voxel[0, 0:-1], patch_factor=2, patch_num=24, stride_factor=2)
        print(f"D_out patches: {patches.shape}")

    def test_get_random_non_cube_triplet_patches(self):
        voxel1 = torch.rand((1, 4, 100, 60, 80))
        voxel2 = torch.rand((1, 4, 80, 70, 80))
        anchors, positives, negatives = get_random_non_cube_triplet_patches(
            voxel1, voxel2, patch_factor=2, stride_factor=1, num_triplets=100)
        print(f"anchors: {anchors.shape}")
        print(f"positives: {positives.shape}")
        print(f"negatives: {negatives.shape}")
        anchors = torch.nn.AdaptiveMaxPool3d(1)(anchors)
        positives = torch.nn.AdaptiveMaxPool3d(1)(positives)
        negatives = torch.nn.AdaptiveMaxPool3d(1)(negatives)
        print(f"anchors: {anchors.shape}")
        anchors = anchors.view(-1, 4)
        print(f"anchors: {anchors.shape}")
        triplet_criterion = torch.nn.TripletMarginLoss(margin=0.1, p=2, reduce=True, reduction='mean')
        loss = triplet_criterion(anchors, positives, negatives,)
        print(f"loss: {loss}")

    def test_get_random_non_cube_pair_patches(self):
        voxel1 = torch.rand((1, 4, 100, 60, 80))
        voxel2 = torch.rand((1, 4, 80, 70, 80))
        patches1, patches2 = get_random_non_cube_pair_patches(
            voxel1, voxel2, patch_factor=2, stride_factor=1, num_pairs=28)
        print(f"patches1: {patches1.shape}")
        print(f"patches2: {patches2.shape}")
        patches1 = torch.nn.AdaptiveMaxPool3d(1)(patches1)
        patches2 = torch.nn.AdaptiveMaxPool3d(1)(patches2)
        print(f"patches1: {patches1.shape}")
        patches1 = patches1.view(-1, 4)
        print(f"patches1: {patches1.shape}")
        dist = torch.nn.functional.pairwise_distance(patches1, patches2, 2)
        loss = torch.mean(dist)
        print(f"loss: {loss}")
