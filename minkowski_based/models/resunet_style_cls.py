from MinkowskiEngine.MinkowskiNonlinearity import MinkowskiReLU
from MinkowskiEngine import SparseTensor

from models.modules.common import avg_pool, ConvType, conv, get_norm, NormType, conv_tr
from models.modules.non_trainable_layers import MinkowskiPoolFeaturesPerComponent
from models.modules.resnet_block import BasicBlock
from models.model import Model


class BaseNetStyleCls(Model):

    EMBEDDING_DIM = 32

    def __init__(self, in_channels, out_channels, config, D=3):

        super(BaseNetStyleCls, self).__init__(in_channels, out_channels, config, D)
        self.network_initialization(in_channels, out_channels, config, D)

    def network_initialization(self, in_channels, out_channels, config, D):
        assert D == 3
        bn_momentum = config.bn_momentum

        self.relu = MinkowskiReLU()

        self.conv1 = conv(in_planes=in_channels,
                          out_planes=64,
                          kernel_size=5,
                          stride=1,
                          dilation=1,
                          bias=True,
                          conv_type=ConvType.SPATIAL_HYPERCUBE,
                          D=D)
        self.conv1bn = get_norm(NormType.BATCH_NORM,
                            n_channels=64,
                            D=D,
                            bn_momentum=bn_momentum)

        # downsampling due to stride
        self.conv2 = conv(in_planes=64,
                          out_planes=128,
                          kernel_size=2,
                          stride=2,
                          dilation=1,
                          bias=True,
                          conv_type=ConvType.SPATIAL_HYPERCUBE,
                          D=D)
        self.conv2bn = get_norm(NormType.BATCH_NORM,
                            n_channels=128,
                            D=D,
                            bn_momentum=bn_momentum)

        # downsampling due to stride
        self.conv3 = conv(in_planes=128,
                          out_planes=256,
                          kernel_size=2,
                          stride=2,
                          dilation=1,
                          conv_type=ConvType.SPATIAL_HYPERCUBE,
                          D=D)
        self.conv3bn = get_norm(NormType.BATCH_NORM,
                                 n_channels=256,
                                 D=D,
                                 bn_momentum=bn_momentum)

        # Upsampling
        self.tr_conv1 = conv_tr(in_planes=256,
                                out_planes=256,
                                kernel_size=2,
                                upsample_stride=2,
                                dilation=1,
                                conv_type=ConvType.SPATIAL_HYPERCUBE,
                                D=D)
        self.tr_conv1_norm = get_norm(norm_type=NormType.BATCH_NORM,
                                      n_channels=256,
                                      D=D,
                                      bn_momentum=bn_momentum)
        # Upsampling
        self.tr_conv2 = conv_tr(in_planes=256,
                                out_planes=256,
                                kernel_size=2,
                                upsample_stride=2,
                                dilation=1,
                                conv_type=ConvType.SPATIAL_HYPERCUBE,
                                D=D)
        self.tr_conv2_norm = get_norm(norm_type=NormType.BATCH_NORM,
                                      n_channels=256,
                                      D=D,
                                      bn_momentum=bn_momentum)

        # Final layers
        self.fc1 = conv(256, 128, kernel_size=2, stride=1, bias=True, D=D)
        self.fc1bn = get_norm(norm_type=NormType.BATCH_NORM,
                              n_channels=128,
                              D=D,
                              bn_momentum=bn_momentum)
        self.fc2 = conv(128, 64, kernel_size=2, stride=1, bias=True, D=D)
        self.fc2bn = get_norm(norm_type=NormType.BATCH_NORM,
                               n_channels=64,
                               D=D,
                               bn_momentum=bn_momentum)

        self.emb = conv(64, self.EMBEDDING_DIM, kernel_size=2, stride=1, bias=True, D=D)
        self.emb_cp = MinkowskiPoolFeaturesPerComponent(self.EMBEDDING_DIM)
        self.final = conv(self.EMBEDDING_DIM, out_channels, kernel_size=2, stride=1, bias=True, D=D)


    def forward(self, x: SparseTensor, components_indices: SparseTensor):
        o1 = self.relu(self.conv1bn(self.conv1(x)))
        o2 = self.relu(self.conv2bn(self.conv2(o1)))
        o3 = self.relu(self.conv3bn(self.conv3(o2)))
        o4 = self.relu(self.tr_conv1_norm(self.tr_conv1(o3)))
        o5 = self.relu(self.tr_conv2_norm(self.tr_conv2(o4)))
        o6 = self.relu(self.fc1bn(self.fc1(o5)))
        o7 = self.relu(self.fc2bn(self.fc2(o6)))
        o8 = self.emb(o7)
        o8pc = self.emb_cp(o8, components_indices)
        o9 = self.final(o8pc)
        return o9


class SmallNetStyleCls(BaseNetStyleCls):
    BLOCK = BasicBlock

