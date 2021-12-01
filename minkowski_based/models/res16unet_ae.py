from models.res16unet import Res16UNetBase, BasicBlock
import MinkowskiEngine as ME
import torch.nn as nn


class Res16UNetAEBase(Res16UNetBase):
    pass
    # def weight_initialization(self):
    #     for m in self.modules():
    #         if isinstance(m, ME.MinkowskiBatchNorm):
    #             nn.init.xavier_uniform_(m.bn.weight)
    #             nn.init.constant_(m.bn.bias, 0)


class Res16UNetAEL8B2(Res16UNetAEBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)
