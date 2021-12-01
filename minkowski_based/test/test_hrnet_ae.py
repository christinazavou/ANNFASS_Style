from unittest import TestCase
from MinkowskiEngine import SparseTensor
import torch

from config import get_config
from lib.utils import get_torch_device

import sys

from models.hrnet_ae import HRNetAE3S3BND64IN16

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch


testargs = ["python test_resunet_ae.py",
            "--stylenet_path", "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/buildnet_reconstruction_splits/ply_100K/split_train_val_test_debug",
            "--dataset", "StylenetXYZAEVoxelization0_01Dataset",
            "--input_feat", "coords"]
with patch.object(sys, 'argv', testargs):
    config = get_config()


class TestHRNetAE3S(TestCase):

    def test_kati(self):
        samples = 100000
        coords = torch.randn(samples, 4).cuda()
        input = torch.randn(samples, 3).cuda()

        device = get_torch_device(True)
        sinput = SparseTensor(input, coords).to(device)

        model = HRNetAE3S3BND64IN16(3, 3, config, 3)
        model = model.to(device)

        soutput = model(sinput)
        assert sinput.F.shape == soutput.F.shape

# INIT
# we have self.inplanes = init_dim (32)
# conv0s1 translates the input channels to inplanes
# cn0s1 applies batch norm
# self.init_stage_dims = init_dim * feat factor
# conv1s1 transforms self.inplanes to self.init_stage_dims (the resolution for the first stage)

# BLOCK is by default a resnet block with two convolutions and batch norm (not bottleneck)

# STAGES
# for stage (i) = 0:
#   stage_layers = []
#   for j in 0...i:
#      (j=0) self.inplanes = self.init_stage_dims * 2 ^ 0
#            add to stage_layers _make_layer(adds a few BLOCKs with initial inplanes=self.inplanes):
#               i.e. after one BLOCK, the self.inplanes becomes equal to output planes of the BLOCK.
#               Thus with a BasicResBlock self.inplanes will remain the same, whereas with a BottleneckBlock
#               with expansion it will become self.inplanes * expansion
#               Note: the following BLOCKs (except the first one) will keep same output channels and will use
#               stride 1
#               so anyway at the end (and start) of these layers we have dimension self.init_stage_dims * 2 ^ 0
#   depth = len(stage) i.e. how many layers (because we except 1 in stage 1, 2 in stage 2...)

#   now if we are not in the last stage, we need to add exchange blocks (i.e. go to layers of different resolutions)
#   so we will loop for each pair (j,k) of existing depths
#   question: are the layers with different resolution in different stages?

# stage=0
#   for j in 0..depth:
#       (j=0) init_channels = self.init_stage_dims * 2 ^ 0
#           for k in 0...depth:
#               d0 = depth-j, d1 = depth-k (the two depths)
#               if d0 > d1:
#                   for s in 0...d0-d1:
#                   we do upsample i.e. add conv with input: init_channels * 2 ^ s and
#                   output: init_channels * 2 ^ (s+1)
#                   where init_channels is the resolution of d0

# let's see for stage = 1:
#   stage_layers = []
#   j = 0:
#       self.inplanes = 32
#       add a stage layer (i.e. two basic blocks i.e. 2 x res-(2conv+2bn)) with 32 in, 32 out
#   j = 1:
#       self.inplanes = 64
#       add a stage layer with (i.e. two basic blocks i.e. 2 x res-(2conv+2bn)) with 64 in, 64 out
#   since we are not at last stage, loop over pairs of depth to add exchange connections:
#   j = 0:
#       init_channels = self.init_stage_dims * 2 ** j = 32
#       k = 0:
#           d0 = 2, d1 = 2:
#               adds no exchange connection
#       k = 1:
#           d0 = 2, d1 = 1:
#               s = 0: add an exchange connection from 32 to 64
#       k = 2:
#           d0 = 2, d1 = 0:
#               s = 0: add an exchange connection from 32 to 64
#               s = 1: add an exchange connection from 64 to 128
#   j = 1:
#       init_channels = self.init_stage_dims * 2 ** j = 64
#       k = 0:
#           d0 = 1, d1 = 2
#               adds an exchange connection from 64 to 32
#       k = 1:
#           d0 = 1, d1 = 1
#               adds no exchange connection
#       k = 2:
#           d0 = 1, d1 = 0
#               s = 0: add an exchange connection from 64 to 128

# FINALLY
# for i in 1..stage-1:
#   add extra convolutions of same in,out channels, and of the resolution of that stage
#   the smaller the resolution the more the convolutions i.e. stage 0 has no extra convolutions,
#   stage 1 has one, stage 2 has two

# backbone_out_feat = np.sum([self.init_stage_dims * 2 ** s for s in range(self.NUM_STAGES)]) + self.INIT_DIM
# thus for 3 stages with initial dim 32, we have 32+64+128 + 32 = 256 backbone_out_feat

# on that backbone_out_feat the default classification in hrnet is to then use a convolution layer
# with 256 as output, and then batch norm, relu, dropout & another conv with out dim the amount of classes


# """
# note:
# out_init : 1620, 32
# stage_input:    0: 1620, 32
#                 1: 379, 64
#                 2: 87, 128
# stage_output:   0: 1620, 32
#                 1: 379, 64
#                 2: 87, 128
# out: [out_init, stage_output[0]]
# & for each subsequent stage:
#     out += [final convolution on relevant stage_output]
#
# out = [32+64+128 + 32] = 1620, 256
# & apply final layer classification layers on that
# """
