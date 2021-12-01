import torch

from MinkowskiEngine.Common import MinkowskiModuleBase
from MinkowskiEngine import SparseTensor


def get_average_per_component_t(c_indices_mat: torch.Tensor, features: torch.Tensor, is_target=False):
    outfeat = torch.matmul(torch.unsqueeze(c_indices_mat.float(), dim=-2), torch.unsqueeze(features, dim=-3).float())
    outfeat = torch.div(torch.squeeze(outfeat, dim=1), torch.unsqueeze(torch.sum(c_indices_mat, 1), dim=-1)).float()
    if is_target:
        assert torch.sum(torch.remainder(outfeat, 1) == 0) == outfeat.numel(), "oops: component with different labeled points"
    return outfeat


class MinkowskiPoolFeaturesPerComponent(MinkowskiModuleBase):

    def __init__(self, channels):
        self.channels = channels
        self.training = False
        super(MinkowskiPoolFeaturesPerComponent, self).__init__()

    def forward(self, input: SparseTensor, c_indices: SparseTensor):
        assert isinstance(input, SparseTensor)
        features = input.F
        assert features.shape[1] == self.channels
        outfeat = get_average_per_component_t(c_indices.F, features)
        return SparseTensor(outfeat, coords=c_indices.C)

    def __repr__(self):
        s = '(channels={}, training={}'.format(self.channels, self.training)
        return self.__class__.__name__ + s
