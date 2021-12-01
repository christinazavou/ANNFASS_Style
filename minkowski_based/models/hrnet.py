import torch.nn as nn
import numpy as np

from MinkowskiEngine import MinkowskiReLU, MinkowskiBatchNorm, MinkowskiDropout
import MinkowskiEngine.MinkowskiOps as me

from models.model import Model
from models.modules.common import ConvType, NormType, get_norm, conv, conv_tr
from models.modules.resnet_block import BasicBlock, Bottleneck


class HRNetBase(Model):
  BLOCK = None
  NUM_STAGES = 1
  NUM_BLOCKS = 3
  INIT_DIM = 32
  FEAT_FACTOR = 1
  OUT_PIXEL_DIST = 1
  NORM_TYPE = NormType.BATCH_NORM
  NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
  CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS
  CLASS_LAYER_DIM = 256

  def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
    assert self.BLOCK is not None

    super(HRNetBase, self).__init__(in_channels, out_channels, config, D, **kwargs)

    self.network_initialization(in_channels, out_channels, config, D)
    self.weight_initialization()

  def network_initialization(self, in_channels, out_channels, config, D):
    # Setup net_metadata
    bn_momentum = config.bn_momentum
    self.export_feat = config.export_feat

    # Initial input features transformation -> concat with hrnet output
    self.inplanes = self.INIT_DIM
    self.conv0s1 = conv(
      in_planes=in_channels,
      out_planes=self.inplanes,
      kernel_size=config.conv1_kernel_size,
      stride=1,
      dilation=1,
      conv_type=self.NON_BLOCK_CONV_TYPE,
      D=D)
    self.bn0s1 = get_norm(norm_type=self.NORM_TYPE, n_channels=self.inplanes, D=D, bn_momentum=bn_momentum)

    # Create high-to-low resolution branches
    self.init_stage_dims = self.INIT_DIM * self.FEAT_FACTOR
    self.conv1s1 = conv(
      in_planes=self.inplanes,
      out_planes=self.init_stage_dims,
      kernel_size=3,
      stride=1,
      dilation=1,
      conv_type=self.NON_BLOCK_CONV_TYPE,
      D=D)
    self.bn1s1 = get_norm(norm_type=self.NORM_TYPE, n_channels=self.init_stage_dims, D=D, bn_momentum=bn_momentum)
    self.stages, self.exchange_blocks = nn.ModuleList([]), nn.ModuleList([])
    for i in range(self.NUM_STAGES):
      # Each stage includes #stage branches
      stage = nn.ModuleList([])
      for j in range(i + 1):
        self.inplanes = self.init_stage_dims * 2 ** j
        stage.append(
          self._make_layer(
            block=self.BLOCK,
            planes=self.init_stage_dims * 2 ** j,
            blocks=self.NUM_BLOCKS,
            stride=1,
            dilation=1,
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum))
      self.stages.append(stage)

      # Create exchange blocks
      if i == (self.NUM_STAGES - 1):
        # No exchange blocks for the last stage
        break
      exchange_blocks = nn.ModuleList([])
      depth = len(stage)
      for j in range(depth):
        exchange_block = nn.ModuleList([])
        init_channels = self.init_stage_dims * 2 ** j
        for k in range(depth + 1):
          d0, d1 = depth - j, depth - k
          block = nn.ModuleList([])
          add_relu = False
          if d0 > d1:
            # Downsampling
            for s in range(d0 - d1):
              if add_relu:
                block.append(MinkowskiReLU())
              block.append(
                conv(
                  in_planes=init_channels * 2 ** s,
                  out_planes=init_channels * 2 ** (s + 1),
                  kernel_size=3,
                  stride=2,
                  dilation=1,
                  conv_type=self.NON_BLOCK_CONV_TYPE,
                  D=D))
              block.append(
                get_norm(norm_type=self.NORM_TYPE, n_channels=init_channels * 2 ** (s + 1),
                         D=D, bn_momentum=bn_momentum))
              add_relu = True
          elif d0 < d1:
            # Upsampling
            for s in range(0, -(d1 - d0), -1):
              if add_relu:
                block.append(MinkowskiReLU())
              block.append(
                conv_tr(
                  in_planes=int(init_channels * 2 ** s),
                  out_planes=int(init_channels * 2 ** (s - 1)),
                  kernel_size=3,
                  upsample_stride=2,
                  dilation=1,
                  conv_type=self.NON_BLOCK_CONV_TYPE,
                  D=D))
              block.append(
                  get_norm(norm_type=self.NORM_TYPE, n_channels=int(init_channels * 2 ** (s - 1)),
                           D=D, bn_momentum=bn_momentum))
              add_relu = True
          else:
            block.append(nn.ModuleList([]))
          exchange_block.append(nn.Sequential(*block))
        exchange_blocks.append(exchange_block)
      self.exchange_blocks.append(exchange_blocks)

    # Final transitions
    self.final_transitions = nn.ModuleList([])
    for i in range(1, self.NUM_STAGES):
      # Upsample all lower resolution branches to the highest resolution branch
      init_channels = self.init_stage_dims * 2 ** i
      block = nn.ModuleList([])
      for j in range(i):
        block.append(
          conv_tr(
            in_planes=init_channels,
            out_planes=init_channels,
            kernel_size=3,
            upsample_stride=2,
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D))
        block.append(
          get_norm(norm_type=self.NORM_TYPE, n_channels=init_channels,
                   D=D, bn_momentum=bn_momentum))
        block.append(MinkowskiReLU())
      self.final_transitions.append(nn.Sequential(*block))

    # FC layer
    backbone_out_feat = np.sum([self.init_stage_dims * 2 ** s for s in range(self.NUM_STAGES)]) + self.INIT_DIM
    self.final_classification_layers(D, backbone_out_feat, bn_momentum, out_channels)

  def final_classification_layers(self, D, backbone_out_feat, bn_momentum, out_channels):
    fc1 = conv(in_planes=backbone_out_feat, out_planes=self.CLASS_LAYER_DIM, kernel_size=1, stride=1, bias=True, D=D)
    bnfc1 = get_norm(norm_type=self.NORM_TYPE, n_channels=self.CLASS_LAYER_DIM, D=D, bn_momentum=bn_momentum)
    dp1 = MinkowskiDropout(0.5)
    fc2 = conv(in_planes=self.CLASS_LAYER_DIM, out_planes=out_channels, kernel_size=1, stride=1, bias=True, D=D)
    self.relu = MinkowskiReLU(inplace=True)
    self.final = nn.Sequential(fc1, bnfc1, self.relu, fc2)
    # self.final = nn.Sequential(fc1, bnfc1, self.relu, dp1, fc2)

  def forward(self, x):
    # Initial input features transformation
    out = self.conv0s1(x)
    out = self.bn0s1(out)
    out_init = self.relu(out)

    # Feature transform to high-resolution branch
    out = self.conv1s1(out_init)
    out = self.bn1s1(out)
    out = self.relu(out)

    # Transform features through HRNet multi-resolution branches
    for i in range(self.NUM_STAGES):
      if i == 0:
        # Only for 1st stage
        stage_input = [out]
      stage_output = []
      for j in range(i + 1):
        stage_output.append(self.stages[i][j](stage_input[j]))
      if i == (self.NUM_STAGES - 1):
        # No exchange blocks for the last stage
        break
      stage_input = [[] for _ in range(len(self.stages[i + 1]))]
      m = len(stage_input)
      depth = len(stage_output)
      for j in range(depth):
        for k in range(depth + 1):
          if j < k:
            # Downsampling
            stage_input[k].append(self.exchange_blocks[i][j][k % m](stage_output[j]))
          elif j > k:
            # Upsampling
            stage_input[k].append(self.exchange_blocks[i][j][k % m](stage_output[j]))
          else:
            stage_input[k].append(stage_output[j])
      for j in range(len(stage_input)):
        buf = stage_input[j][0]
        for k in range(1, len(stage_input[j])):
          buf = buf + stage_input[j][k]
        stage_input[j] = self.relu(buf)

    # Final transitions
    out = [out_init, stage_output[0]]
    for i in range(1, self.NUM_STAGES):
      out.append(self.final_transitions[i - 1](stage_output[i]))

    _out = me.cat(*out)

    if self.export_feat:
      if len(out) == 2:
        return self.final(_out), out[0], out[1], None
      return self.final(_out), out[1], out[2], out[3]

    return self.final(_out)

  def weight_initialization(self):
    for m in self.modules():
      if isinstance(m, MinkowskiBatchNorm):
        nn.init.constant_(m.bn.weight, 1)
        nn.init.constant_(m.bn.bias, 0)

  def _make_layer(self,
                  block,
                  planes,
                  blocks,
                  stride=1,
                  dilation=1,
                  norm_type=NormType.BATCH_NORM,
                  bn_momentum=0.1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          conv(
              self.inplanes,
              planes * block.expansion,
              kernel_size=1,
              stride=stride,
              bias=False,
              D=self.D),
          get_norm(norm_type, planes * block.expansion, D=self.D, bn_momentum=bn_momentum)
      )
    layers = []
    layers.append(
        block(
            self.inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            conv_type=self.CONV_TYPE,
            D=self.D,
            bn_momentum=bn_momentum))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(
          block(
              self.inplanes,
              planes,
              stride=1,
              dilation=dilation,
              conv_type=self.CONV_TYPE,
              D=self.D,
              bn_momentum=bn_momentum))

    return nn.Sequential(*layers)


class HRNet2S(HRNetBase):
  BLOCK = BasicBlock
  FEAT_FACTOR = 4
  NUM_STAGES = 2


class HRNet3S(HRNetBase):
  BLOCK = BasicBlock
  FEAT_FACTOR = 4
  NUM_STAGES = 3


class HRNet3S2BD256(HRNetBase):
  BLOCK = BasicBlock
  FEAT_FACTOR = 1
  NUM_STAGES = 3
  NUM_BLOCKS = 2


class HRNet4S(HRNetBase):
  BLOCK = BasicBlock
  FEAT_FACTOR = 2
  NUM_STAGES = 4


class HRNet1S2BD128(HRNetBase):
    BLOCK = BasicBlock
    FEAT_FACTOR = 1
    NUM_STAGES = 1
    NUM_BLOCKS = 3
    CLASS_LAYER_DIM = 128


class HRNet3S3BND256(HRNetBase):
    BLOCK = Bottleneck
    FEAT_FACTOR = 1
    NUM_STAGES = 3
    NUM_BLOCKS = 2


class HRNet3S2BND128(HRNetBase):
    BLOCK = Bottleneck
    FEAT_FACTOR = 1
    NUM_STAGES = 3
    NUM_BLOCKS = 2
    CLASS_LAYER_DIM = 128
