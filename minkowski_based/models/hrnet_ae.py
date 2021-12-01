from models.hrnet import HRNetBase
from models.modules.resnet_block import BasicBlock, Bottleneck, Bottleneck2
import MinkowskiEngine as ME
import torch.nn as nn
import MinkowskiEngine.MinkowskiOps as me


class HRNetAEBase(HRNetBase):
    pass
    # def weight_initialization(self):
    #     for m in self.modules():
    #         if isinstance(m, ME.MinkowskiBatchNorm):
    #             nn.init.xavier_uniform_(m.bn.weight)
    #             nn.init.constant_(m.bn.bias, 0)

    def log_weights(self):
        weights = {}
        weights['conv0s1'] = (self.conv0s1.kernel.data.min(), self.conv0s1.kernel.data.max())
        weights['conv1s1'] = (self.conv1s1.kernel.data.min(), self.conv1s1.kernel.data.max())
        return weights

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


class HRNetAE3S2BD128(HRNetAEBase):
    BLOCK = BasicBlock
    FEAT_FACTOR = 1
    NUM_STAGES = 3
    NUM_BLOCKS = 2
    CLASS_LAYER_DIM = 128


class HRNetAE3S2BD256(HRNetAEBase):
    BLOCK = BasicBlock
    FEAT_FACTOR = 1
    NUM_STAGES = 3
    NUM_BLOCKS = 2


class HRNetAE1S2BD128(HRNetAEBase):
    BLOCK = BasicBlock
    FEAT_FACTOR = 1
    NUM_STAGES = 1
    NUM_BLOCKS = 3
    CLASS_LAYER_DIM = 128


class HRNetAE3S3BND256(HRNetAEBase):
    BLOCK = Bottleneck
    FEAT_FACTOR = 1
    NUM_STAGES = 3
    NUM_BLOCKS = 2


class HRNetAE3S3BNDF4256(HRNetAEBase):
    BLOCK = Bottleneck
    FEAT_FACTOR = 4
    NUM_STAGES = 3
    NUM_BLOCKS = 3


class HRNetAE3S2BND128(HRNetAEBase):
    BLOCK = Bottleneck
    FEAT_FACTOR = 1
    NUM_STAGES = 3
    NUM_BLOCKS = 2
    CLASS_LAYER_DIM = 128


class HRNetAE3S3BD64IN8(HRNetAEBase):
    BLOCK = BasicBlock
    FEAT_FACTOR = 1
    NUM_STAGES = 3
    NUM_BLOCKS = 3
    CLASS_LAYER_DIM = 64
    INIT_DIM = 8


class HRNetAE3S3BD64IN16(HRNetAEBase):
    BLOCK = BasicBlock
    FEAT_FACTOR = 1
    NUM_STAGES = 3
    NUM_BLOCKS = 3
    CLASS_LAYER_DIM = 64
    INIT_DIM = 16


class HRNetAE3S3BND64IN16(HRNetAEBase):
    BLOCK = Bottleneck2
    FEAT_FACTOR = 1
    NUM_STAGES = 3
    NUM_BLOCKS = 3
    CLASS_LAYER_DIM = 64
    INIT_DIM = 16


