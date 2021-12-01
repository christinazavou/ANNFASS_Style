import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable


def initialize_conv_relu_layers(m):  #assume all conv layers of m use relu / leaky relu
    for module in m.modules():
        if isinstance(module, nn.Conv3d):
            # nn.init.xavier_normal_(m.weight)  # for sigmoid / tanh activation
            nn.init.kaiming_normal_(module.weight)  # for relu / leaky rely activation
            if getattr(module, 'bias', None) is not None:
                nn.init.constant_(module.bias, 0)

#cell = 4
#input 256
#output 120 (128-4-4)
#receptive field = 18

#            0  18
#conv 4x4 s1 4  15
#conv 3x3 s2 6  7
#conv 3x3 s1 10 5
#conv 3x3 s1 14 3
#conv 3x3 s1 18 1
#conv 1x1 s1 1  1
class discriminator(nn.Module):
    def __init__(self, d_dim, z_dim, wasserstein=False, init_weights=False):
        super(discriminator, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.wasserstein = wasserstein

        self.conv_1 = nn.Conv3d(1,             self.d_dim,    4, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=0, bias=True)
        self.conv_3 = nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.d_dim*4,  self.d_dim*8,  3, stride=1, padding=0, bias=True)
        self.conv_5 = nn.Conv3d(self.d_dim*8,  self.d_dim*16, 3, stride=1, padding=0, bias=True)
        self.conv_6 = nn.Conv3d(self.d_dim*16, self.z_dim,    1, stride=1, padding=0, bias=True)

        self.pool = nn.AdaptiveMaxPool3d(1)

        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            self.initialize_weights()

    def forward(self, voxels, is_training=False):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_6(out)
        if not self.wasserstein:
            out = torch.sigmoid(out)

        return out

    def encode(self, voxels, is_training=False):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out1 = out

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out2 = out

        out = self.conv_6(out)
        if not self.wasserstein:
            out = torch.sigmoid(out)

        return out, out2, out1

    def layer(self, voxels, layer='all'):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out3 = out
        if layer == 'c3act':
            return self.pool(out).squeeze(2).squeeze(2).squeeze(2)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out4 = out
        if layer == 'c4act':
            return self.pool(out).squeeze(2).squeeze(2).squeeze(2)

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out5 = out
        if layer == 'c5act':
            return self.pool(out).squeeze(2).squeeze(2).squeeze(2)

        out = self.conv_6(out)
        if not self.wasserstein:
            out = torch.sigmoid(out)
        if layer == 'last':
            return self.pool(out).squeeze(2).squeeze(2).squeeze(2)

        if layer == 'all':
            out = torch.cat([self.pool(out).squeeze(2).squeeze(2).squeeze(2).T,
                             self.pool(out5).squeeze(2).squeeze(2).squeeze(2).T,
                             self.pool(out4).squeeze(2).squeeze(2).squeeze(2).T,
                             self.pool(out3).squeeze(2).squeeze(2).squeeze(2).T]).T
            return out

    def initialize_weights(self):
        for conv in [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5]:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.constant_(conv.bias, 0)
        if not self.wasserstein:
            nn.init.xavier_normal_(self.conv_6.weight)
            nn.init.constant_(self.conv_6.bias, 0)
        else:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.constant_(conv.bias, 0)


class discriminator_wgp(nn.Module):
    def __init__(self, d_dim, z_dim, activation="relu", init_weights=False):
        super(discriminator_wgp, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.activation = activation

        self.conv_1 = nn.Conv3d(1,             self.d_dim,    4, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=0, bias=True)
        self.conv_3 = nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.d_dim*4,  self.d_dim*8,  3, stride=1, padding=0, bias=True)
        self.conv_5 = nn.Conv3d(self.d_dim*8,  self.d_dim*16, 3, stride=1, padding=0, bias=True)
        self.conv_6 = nn.Conv3d(self.d_dim*16, self.z_dim,    1, stride=1, padding=0, bias=True)

        self.pool = nn.AdaptiveMaxPool3d(1)

        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            self.initialize_weights()

    def forward(self, voxels, is_training=False):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_6(out)
        if self.activation == 'sigmoid':
            out = torch.sigmoid(out)
        elif self.activation == 'relu':
            out = F.relu(out)
        else:
            raise Exception("unknown activation in D")

        return out

    def encode(self, voxels, is_training=False):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out1 = out

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out2 = out

        out = self.conv_6(out)
        if self.activation == 'sigmoid':
            out = torch.sigmoid(out)
        else:
            out = F.relu(out)

        return out, out2, out1

    def layer(self, voxels, layer='all'):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out3 = out
        if layer == 'c3act':
            return self.pool(out).squeeze(2).squeeze(2).squeeze(2)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out4 = out
        if layer == 'c4act':
            return self.pool(out).squeeze(2).squeeze(2).squeeze(2)

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out5 = out
        if layer == 'c5act':
            return self.pool(out).squeeze(2).squeeze(2).squeeze(2)

        out = self.conv_6(out)
        if self.activation == 'sigmoid':
            out = torch.sigmoid(out)
        else:
            out = F.relu(out)
        if layer == 'last':
            return self.pool(out).squeeze(2).squeeze(2).squeeze(2)

        if layer == 'all':
            out = torch.cat([self.pool(out).squeeze(2).squeeze(2).squeeze(2).T,
                             self.pool(out5).squeeze(2).squeeze(2).squeeze(2).T,
                             self.pool(out4).squeeze(2).squeeze(2).squeeze(2).T,
                             self.pool(out3).squeeze(2).squeeze(2).squeeze(2).T]).T
            return out

    def initialize_weights(self):
        for conv in [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5]:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.constant_(conv.bias, 0)
        if self.activation == 'sigmoid':
            nn.init.xavier_normal_(self.conv_6.weight)
            nn.init.constant_(self.conv_6.bias, 0)
        else:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.constant_(conv.bias, 0)


class common_discriminator(nn.Module):
    def __init__(self, d_dim, init_weights=False):
        super(common_discriminator, self).__init__()
        self.d_dim = d_dim

        self.conv_1 = nn.Conv3d(1,             self.d_dim,    4, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=0, bias=True)
        self.conv_3 = nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.d_dim*4,  self.d_dim*8,  3, stride=1, padding=0, bias=True)
        self.conv_5 = nn.Conv3d(self.d_dim*8,  self.d_dim*16, 3, stride=1, padding=0, bias=True)

        self.pool = nn.AdaptiveMaxPool3d(1)

        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)

    def forward(self, voxels, is_training=False):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        return out


class common_discriminator_1(nn.Module):
    def __init__(self, d_dim, init_weights=False):
        super(common_discriminator_1, self).__init__()
        self.d_dim = d_dim
        self.conv_1 = nn.Conv3d(1,             self.d_dim,    4, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=0, bias=True)
        self.conv_3 = nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.d_dim*4,  self.d_dim*8,  3, stride=1, padding=0, bias=True)
        self.pool = nn.AdaptiveMaxPool3d(1)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)

    def forward(self, voxels, is_training=False):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        return out


def gn_init(m, zero_init=False):
    assert isinstance(m, nn.GroupNorm)
    m.weight.data.fill_(0. if zero_init else 1.)
    m.bias.data.zero_()


class common_discriminator_1_gn(nn.Module):
    def __init__(self, d_dim, init_weights=False):
        super(common_discriminator_1_gn, self).__init__()
        self.d_dim = d_dim
        self.conv_1 = nn.Conv3d(1,             self.d_dim,    4, stride=1, padding=0, bias=True)
        self.gn_1 = nn.GroupNorm(1, self.d_dim)
        self.conv_2 = nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=0, bias=True)
        self.gn_2 = nn.GroupNorm(2, self.d_dim*2)
        self.conv_3 = nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=1, padding=0, bias=True)
        self.gn_3 = nn.GroupNorm(4, self.d_dim*4)
        self.conv_4 = nn.Conv3d(self.d_dim*4,  self.d_dim*8,  3, stride=1, padding=0, bias=True)
        self.gn_4 = nn.GroupNorm(8, self.d_dim*8)
        self.pool = nn.AdaptiveMaxPool3d(1)
        gn_init(self.gn_1)
        gn_init(self.gn_2)
        gn_init(self.gn_3)
        gn_init(self.gn_4, zero_init=True)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)

    def forward(self, voxels, is_training=False):
        out = voxels

        out = self.gn_1(self.conv_1(out))
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.gn_2(self.conv_2(out))
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.gn_3(self.conv_3(out))
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.gn_4(self.conv_4(out))
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        return out


class common_discriminator_2(nn.Module):
    def __init__(self, d_dim, init_weights=False):
        super(common_discriminator_2, self).__init__()
        self.d_dim = d_dim
        self.conv_1 = nn.Conv3d(1,             self.d_dim,    4, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=0, bias=True)
        self.conv_3 = nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.d_dim*4,  self.d_dim*8,  3, stride=1, padding=0, bias=True)
        self.conv_5 = nn.Conv3d(self.d_dim*8,  self.d_dim*16, 3, stride=1, padding=0, bias=True)
        self.pool = nn.AdaptiveMaxPool3d(1)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)

    def forward(self, voxels, is_training=False):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        return out


class common_discriminator_2_gn(nn.Module):
    def __init__(self, d_dim, init_weights=False):
        super(common_discriminator_2_gn, self).__init__()
        self.d_dim = d_dim
        self.conv_1 = nn.Conv3d(1,             self.d_dim,    4, stride=1, padding=0, bias=True)
        self.gn_1 = nn.GroupNorm(1, self.d_dim)
        self.conv_2 = nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=0, bias=True)
        self.gn_2 = nn.GroupNorm(2, self.d_dim*2)
        self.conv_3 = nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=1, padding=0, bias=True)
        self.gn_3 = nn.GroupNorm(4, self.d_dim*4)
        self.conv_4 = nn.Conv3d(self.d_dim*4,  self.d_dim*8,  3, stride=1, padding=0, bias=True)
        self.gn_4 = nn.GroupNorm(8, self.d_dim*8)
        self.conv_5 = nn.Conv3d(self.d_dim*8,  self.d_dim*16, 3, stride=1, padding=0, bias=True)
        self.gn_5 = nn.GroupNorm(16, self.d_dim*16)
        self.pool = nn.AdaptiveMaxPool3d(1)
        gn_init(self.gn_1)
        gn_init(self.gn_2)
        gn_init(self.gn_3)
        gn_init(self.gn_4)
        gn_init(self.gn_5, zero_init=True)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)

    def forward(self, voxels, is_training=False):
        out = voxels

        out = self.gn_1(self.conv_1(out))
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.gn_2(self.conv_2(out))
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.gn_3(self.conv_3(out))
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.gn_4(self.conv_4(out))
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.gn_5(self.conv_5(out))
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        return out


class common_discriminator_3(nn.Module):
    def __init__(self, d_dim, init_weights=False):
        super(common_discriminator_3, self).__init__()
        self.d_dim = d_dim
        self.conv_1 = nn.Conv3d(1,             self.d_dim,    4, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=0, bias=True)
        self.conv_3 = nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=1, padding=0, bias=True)
        self.pool = nn.AdaptiveMaxPool3d(1)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        return out


class common_discriminator_3_gn(nn.Module):

    def __init__(self, d_dim, init_weights=False):
        super(common_discriminator_3_gn, self).__init__()
        self.d_dim = d_dim
        self.conv_1 = nn.Conv3d(1,             self.d_dim,    4, stride=1, padding=0, bias=True)
        self.gn_1 = nn.GroupNorm(1, self.d_dim)
        self.conv_2 = nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=0, bias=True)
        self.gn_2 = nn.GroupNorm(2, self.d_dim*2)
        self.conv_3 = nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=1, padding=0, bias=True)
        self.gn_3 = nn.GroupNorm(4, self.d_dim*4)
        self.pool = nn.AdaptiveMaxPool3d(1)
        gn_init(self.gn_1)
        gn_init(self.gn_2)
        gn_init(self.gn_3, zero_init=True)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.gn_1(self.conv_1(out))
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out = self.gn_2(self.conv_2(out))
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out = self.gn_3(self.conv_3(out))
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        return out


class common_discriminator_4(nn.Module):
    def __init__(self, d_dim, init_weights=False):
        super(common_discriminator_4, self).__init__()
        self.d_dim = d_dim
        self.conv_1 = nn.Conv3d(1,             self.d_dim,    4, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=0, bias=True)
        self.pool = nn.AdaptiveMaxPool3d(1)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        return out


class common_discriminator_4_gn(nn.Module):
    def __init__(self, d_dim, init_weights=False):
        super(common_discriminator_4_gn, self).__init__()
        self.d_dim = d_dim
        self.conv_1 = nn.Conv3d(1,             self.d_dim,    4, stride=1, padding=0, bias=True)
        self.gn_1 = nn.GroupNorm(1, self.d_dim)
        self.conv_2 = nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=0, bias=True)
        self.gn_2 = nn.GroupNorm(2, self.d_dim*2)
        self.pool = nn.AdaptiveMaxPool3d(1)
        gn_init(self.gn_1)
        gn_init(self.gn_2, zero_init=True)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.gn_1(self.conv_1(out))
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out = self.gn_2(self.conv_2(out))
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        return out


class discriminator_part_global_plausibility_1(nn.Module):
    def __init__(self, d_dim, wasserstein=False, init_weights=False):
        super(discriminator_part_global_plausibility_1, self).__init__()
        self.d_dim = d_dim
        self.conv_5 = nn.Conv3d(self.d_dim*8,  self.d_dim*16, 3, stride=1, padding=0, bias=True)
        self.conv_global_discr = nn.Conv3d(self.d_dim*16, 1,    1, stride=1, padding=0, bias=True)
        self.pool = nn.AdaptiveMaxPool3d(1)
        self.wasserstein = wasserstein
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            self.initialize_weights()

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.conv_5(out)
        out = torch.relu(out)
        out = self.conv_global_discr(out)
        if not self.wasserstein:
            out = torch.sigmoid(out)
        return out

    def layer(self, voxels,):
        out = voxels
        out = self.conv_5(out)
        out = torch.relu(out)
        out = self.pool(out).squeeze(2).squeeze(2).squeeze(2)
        return out

    def initialize_weights(self):
        nn.init.kaiming_normal_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias, 0)
        if not self.wasserstein:
            nn.init.xavier_normal_(self.conv_global_discr.weight)
        else:
            nn.init.kaiming_normal_(self.conv_global_discr.weight)
        nn.init.constant_(self.conv_global_discr.bias, 0)


class discriminator_part_global_plausibility_1_gn(nn.Module):
    def __init__(self, d_dim, wasserstein=False, init_weights=False):
        super(discriminator_part_global_plausibility_1_gn, self).__init__()
        self.d_dim = d_dim
        self.conv_5 = nn.Conv3d(self.d_dim*8,  self.d_dim*16, 3, stride=1, padding=0, bias=True)
        self.gn_5 = nn.GroupNorm(16, self.d_dim*16)
        self.conv_global_discr = nn.Conv3d(self.d_dim*16, 1,    1, stride=1, padding=0, bias=True)
        self.pool = nn.AdaptiveMaxPool3d(1)
        self.wasserstein = wasserstein
        gn_init(self.gn_5, zero_init=True)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            self.initialize_weights()

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.gn_5(self.conv_5(out))
        out = torch.relu(out)
        out = self.conv_global_discr(out)
        if not self.wasserstein:
            out = torch.sigmoid(out)
        return out

    def layer(self, voxels,):
        out = voxels
        out = self.gn_5(self.conv_5(out))
        out = torch.relu(out)
        out = self.pool(out).squeeze(2).squeeze(2).squeeze(2)
        return out

    def initialize_weights(self):
        nn.init.kaiming_normal_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias, 0)
        if not self.wasserstein:
            nn.init.xavier_normal_(self.conv_global_discr.weight)
        else:
            nn.init.kaiming_normal_(self.conv_global_discr.weight)
        nn.init.constant_(self.conv_global_discr.bias, 0)


class discriminator_part_global_plausibility_3(nn.Module):
    def __init__(self, d_dim, wasserstein=False, init_weights=False):
        super(discriminator_part_global_plausibility_3, self).__init__()
        self.d_dim = d_dim
        self.conv_4 = nn.Conv3d(self.d_dim*4,  self.d_dim*8, 3, stride=1, padding=0, bias=True)
        self.conv_5 = nn.Conv3d(self.d_dim*8,  self.d_dim*16, 3, stride=1, padding=0, bias=True)
        self.conv_global_discr = nn.Conv3d(self.d_dim*16, 1,    1, stride=1, padding=0, bias=True)
        self.pool = nn.AdaptiveMaxPool3d(1)
        self.wasserstein = wasserstein
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            self.initialize_weights()

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.conv_4(out)
        out = torch.relu(out)
        out = self.conv_5(out)
        out = torch.relu(out)
        out = self.conv_global_discr(out)
        if not self.wasserstein:
            out = torch.sigmoid(out)
        return out

    def layer(self, voxels, is_training=False):
        out = voxels
        out4 = self.conv_4(out)
        out4 = torch.relu(out4)
        out5 = self.conv_5(out4)
        out5 = torch.relu(out5)
        out = self.conv_global_discr(out5)
        if not self.wasserstein:
            out = torch.sigmoid(out)
        out = torch.cat([self.pool(out5).squeeze(2).squeeze(2).squeeze(2).T,
                         self.pool(out4).squeeze(2).squeeze(2).squeeze(2).T,
                         self.pool(out).squeeze(2).squeeze(2).squeeze(2).T]).T
        return out

    def initialize_weights(self):
        nn.init.kaiming_normal_(self.conv_4.weight)
        nn.init.constant_(self.conv_4.bias, 0)
        nn.init.kaiming_normal_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias, 0)
        if not self.wasserstein:
            nn.init.xavier_normal_(self.conv_global_discr.weight)
        else:
            nn.init.kaiming_normal_(self.conv_global_discr.weight)
        nn.init.constant_(self.conv_global_discr.bias, 0)


class discriminator_part_global_plausibility_3_gn(nn.Module):
    def __init__(self, d_dim, wasserstein=False, init_weights=False):
        super(discriminator_part_global_plausibility_3_gn, self).__init__()
        self.d_dim = d_dim
        self.conv_4 = nn.Conv3d(self.d_dim*4,  self.d_dim*8, 3, stride=1, padding=0, bias=True)
        self.gn_4 = nn.GroupNorm(8, self.d_dim*8)
        self.conv_5 = nn.Conv3d(self.d_dim*8,  self.d_dim*16, 3, stride=1, padding=0, bias=True)
        self.gn_5 = nn.GroupNorm(16, self.d_dim*16)
        self.conv_global_discr = nn.Conv3d(self.d_dim*16, 1,    1, stride=1, padding=0, bias=True)
        self.pool = nn.AdaptiveMaxPool3d(1)
        self.wasserstein = wasserstein
        gn_init(self.gn_4)
        gn_init(self.gn_5, zero_init=True)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            self.initialize_weights()

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.gn_4(self.conv_4(out))
        out = torch.relu(out)
        out = self.gn_5(self.conv_5(out))
        out = torch.relu(out)
        out = self.conv_global_discr(out)
        if not self.wasserstein:
            out = torch.sigmoid(out)
        return out

    def layer(self, voxels, is_training=False):
        out = voxels
        out4 = self.gn_4(self.conv_4(out))
        out4 = torch.relu(out4)
        out5 = self.gn_5(self.conv_5(out4))
        out5 = torch.relu(out5)
        out = self.conv_global_discr(out5)
        if not self.wasserstein:
            out = torch.sigmoid(out)
        out = torch.cat([self.pool(out5).squeeze(2).squeeze(2).squeeze(2).T,
                         self.pool(out4).squeeze(2).squeeze(2).squeeze(2).T,
                         self.pool(out).squeeze(2).squeeze(2).squeeze(2).T]).T
        return out

    def initialize_weights(self):
        nn.init.kaiming_normal_(self.conv_4.weight)
        nn.init.constant_(self.conv_4.bias, 0)
        nn.init.kaiming_normal_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias, 0)
        if not self.wasserstein:
            nn.init.xavier_normal_(self.conv_global_discr.weight)
        else:
            nn.init.kaiming_normal_(self.conv_global_discr.weight)
        nn.init.constant_(self.conv_global_discr.bias, 0)


class discriminator_part_global_plausibility_4(nn.Module):
    def __init__(self, d_dim, wasserstein=False, init_weights=False):
        super(discriminator_part_global_plausibility_4, self).__init__()
        self.d_dim = d_dim
        self.conv_3 = nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.d_dim*4,  self.d_dim*8, 3, stride=1, padding=0, bias=True)
        self.conv_5 = nn.Conv3d(self.d_dim*8,  self.d_dim*16, 3, stride=1, padding=0, bias=True)
        self.conv_global_discr = nn.Conv3d(self.d_dim*16, 1,    1, stride=1, padding=0, bias=True)
        self.pool = nn.AdaptiveMaxPool3d(1)
        self.wasserstein = wasserstein
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            self.initialize_weights()

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out = self.conv_global_discr(out)
        if not self.wasserstein:
            out = torch.sigmoid(out)
        return out

    def layer(self, voxels, is_training=False):
        out = voxels
        out3 = self.conv_3(out)
        out3 = F.leaky_relu(out3, negative_slope=0.02, inplace=True)
        out4 = self.conv_4(out3)
        out4 = F.leaky_relu(out4, negative_slope=0.02, inplace=True)
        out5 = self.conv_5(out4)
        out5 = F.leaky_relu(out5, negative_slope=0.02, inplace=True)
        out = self.conv_global_discr(out5)
        if not self.wasserstein:
            out = torch.sigmoid(out)
        out = torch.cat([self.pool(out5).squeeze(2).squeeze(2).squeeze(2).T,
                         self.pool(out4).squeeze(2).squeeze(2).squeeze(2).T,
                         self.pool(out).squeeze(2).squeeze(2).squeeze(2).T]).T
        return out

    def initialize_weights(self):
        nn.init.kaiming_normal_(self.conv_3.weight)
        nn.init.constant_(self.conv_3.bias, 0)
        nn.init.kaiming_normal_(self.conv_4.weight)
        nn.init.constant_(self.conv_4.bias, 0)
        nn.init.kaiming_normal_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias, 0)
        if not self.wasserstein:
            nn.init.xavier_normal_(self.conv_global_discr.weight)
        else:
            nn.init.kaiming_normal_(self.conv_global_discr.weight)
        nn.init.constant_(self.conv_global_discr.bias, 0)


class discriminator_part_global_plausibility_4_gn(nn.Module):
    def __init__(self, d_dim, wasserstein=False, init_weights=False):
        super(discriminator_part_global_plausibility_4_gn, self).__init__()
        self.d_dim = d_dim
        self.conv_3 = nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=1, padding=0, bias=True)
        self.gn_3 = nn.GroupNorm(4, self.d_dim*4)
        self.conv_4 = nn.Conv3d(self.d_dim*4,  self.d_dim*8, 3, stride=1, padding=0, bias=True)
        self.gn_4 = nn.GroupNorm(8, self.d_dim*8)
        self.conv_5 = nn.Conv3d(self.d_dim*8,  self.d_dim*16, 3, stride=1, padding=0, bias=True)
        self.gn_5 = nn.GroupNorm(16, self.d_dim*16)
        self.conv_global_discr = nn.Conv3d(self.d_dim*16, 1,    1, stride=1, padding=0, bias=True)
        self.pool = nn.AdaptiveMaxPool3d(1)
        self.wasserstein = wasserstein
        gn_init(self.gn_3)
        gn_init(self.gn_4)
        gn_init(self.gn_5, zero_init=True)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            self.initialize_weights()

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.gn_3(self.conv_3(out))
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out = self.gn_4(self.conv_4(out))
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out = self.gn_5(self.conv_5(out))
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out = self.conv_global_discr(out)
        if not self.wasserstein:
            out = torch.sigmoid(out)
        return out

    def layer(self, voxels, is_training=False):
        out = voxels
        out3 = self.gn_3(self.conv_3(out))
        out3 = F.leaky_relu(out3, negative_slope=0.02, inplace=True)
        out4 = self.gn_4(self.conv_4(out3))
        out4 = F.leaky_relu(out4, negative_slope=0.02, inplace=True)
        out5 = self.gn_5(self.conv_5(out4))
        out5 = F.leaky_relu(out5, negative_slope=0.02, inplace=True)
        out = self.conv_global_discr(out5)
        if not self.wasserstein:
            out = torch.sigmoid(out)
        out = torch.cat([self.pool(out5).squeeze(2).squeeze(2).squeeze(2).T,
                         self.pool(out4).squeeze(2).squeeze(2).squeeze(2).T,
                         self.pool(out).squeeze(2).squeeze(2).squeeze(2).T]).T
        return out

    def initialize_weights(self):
        nn.init.kaiming_normal_(self.conv_3.weight)
        nn.init.constant_(self.conv_3.bias, 0)
        nn.init.kaiming_normal_(self.conv_4.weight)
        nn.init.constant_(self.conv_4.bias, 0)
        nn.init.kaiming_normal_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias, 0)
        if not self.wasserstein:
            nn.init.xavier_normal_(self.conv_global_discr.weight)
        else:
            nn.init.kaiming_normal_(self.conv_global_discr.weight)
        nn.init.constant_(self.conv_global_discr.bias, 0)


class discriminator_part_global_plausibility_2(nn.Module):
    def __init__(self, d_dim, wasserstein=False, init_weights=False):
        super(discriminator_part_global_plausibility_2, self).__init__()
        self.d_dim = d_dim
        self.conv_global_discr = nn.Conv3d(self.d_dim*16, 1,    1, stride=1, padding=0, bias=True)
        self.pool = nn.AdaptiveMaxPool3d(1)
        self.wasserstein = wasserstein
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            self.initialize_weights()

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.conv_global_discr(out)
        if not self.wasserstein:
            out = torch.sigmoid(out)
        return out

    def initialize_weights(self):
        if not self.wasserstein:
            nn.init.xavier_normal_(self.conv_global_discr.weight)
        else:
            nn.init.kaiming_normal_(self.conv_global_discr.weight)
        nn.init.constant_(self.conv_global_discr.bias, 0)


class discriminator_part_style_plausibility_1(nn.Module):
    def __init__(self, d_dim, z_dim, init_weights=False):
        super(discriminator_part_style_plausibility_1, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.conv_5 = nn.Conv3d(self.d_dim*8,  self.d_dim*16, 3, stride=1, padding=0, bias=True)
        self.conv_style_discr = nn.Conv3d(self.d_dim*16, self.z_dim,    1, stride=1, padding=0, bias=True)
        self.pool = nn.AdaptiveMaxPool3d(1)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.conv_5(out)
        out = torch.relu(out)
        out = self.conv_style_discr(out)
        out = torch.relu(out)
        out = F.normalize(out, p=2, dim=1)
        return out

    def layer(self, voxels, layer='all'):
        out = voxels
        out5 = self.conv_5(out)
        out5 = torch.relu(out5)
        out = self.conv_style_discr(out5)
        out = torch.relu(out)
        out = F.normalize(out, p=2, dim=1)
        if layer == 'all':
            out = torch.cat([self.pool(out).squeeze(2).squeeze(2).squeeze(2).T,
                             self.pool(out5).squeeze(2).squeeze(2).squeeze(2).T]).T
            return out
        else:
            return self.pool(out).squeeze(2).squeeze(2).squeeze(2)


class discriminator_part_style_plausibility_1_gn(nn.Module):
    def __init__(self, d_dim, z_dim, init_weights=False):
        super(discriminator_part_style_plausibility_1_gn, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.conv_5 = nn.Conv3d(self.d_dim*8,  self.d_dim*16, 3, stride=1, padding=0, bias=True)
        self.gn_5 = nn.GroupNorm(16, self.d_dim*16)
        self.conv_style_discr = nn.Conv3d(self.d_dim*16, self.z_dim,    1, stride=1, padding=0, bias=True)
        self.gn_style_discr = nn.GroupNorm(1, self.z_dim)  # equivalent to Layer Normalization
        self.pool = nn.AdaptiveMaxPool3d(1)
        gn_init(self.gn_5)
        gn_init(self.gn_6, zero_init=True)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.gn_5(self.conv_5(out))
        out = torch.relu(out)
        out = self.gn_style_discr(self.conv_style_discr(out))
        out = torch.relu(out)
        # out = F.normalize(out, p=2, dim=1)  # already using layer normalization
        return out

    def layer(self, voxels, layer='all'):
        out = voxels
        out5 = self.gn_5(self.conv_5(out))
        out5 = torch.relu(out5)
        out = self.gn_style_discr(self.conv_style_discr(out5))
        out = torch.relu(out)
        # out = F.normalize(out, p=2, dim=1)  # already using layer normalization
        if layer == 'all':
            out = torch.cat([self.pool(out).squeeze(2).squeeze(2).squeeze(2).T,
                             self.pool(out5).squeeze(2).squeeze(2).squeeze(2).T]).T
            return out
        else:
            return self.pool(out).squeeze(2).squeeze(2).squeeze(2)


class discriminator_part_style_plausibility_3(nn.Module):
    def __init__(self, d_dim, z_dim, init_weights=False):
        super(discriminator_part_style_plausibility_3, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.conv_4 = nn.Conv3d(self.d_dim*4,  self.d_dim*8, 3, stride=1, padding=0, bias=True)
        self.conv_style_discr = nn.Conv3d(self.d_dim*8, self.z_dim,    1, stride=1, padding=0, bias=True)
        self.pool = nn.AdaptiveMaxPool3d(1)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.conv_4(out)
        out = torch.relu(out)
        out = self.conv_style_discr(out)
        out = torch.relu(out)
        out = F.normalize(out, p=2, dim=1)
        return out

    def layer(self, voxels, ):
        out = voxels
        out4 = self.conv_4(out)
        out4 = torch.relu(out4)
        out = self.conv_style_discr(out4)
        out = torch.relu(out)
        out = F.normalize(out, p=2, dim=1)
        out = torch.cat([self.pool(out).squeeze(2).squeeze(2).squeeze(2).T,
                         self.pool(out4).squeeze(2).squeeze(2).squeeze(2).T]).T
        return out


class discriminator_part_style_plausibility_3_gn(nn.Module):
    def __init__(self, d_dim, z_dim, init_weights=False):
        super(discriminator_part_style_plausibility_3_gn, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.conv_4 = nn.Conv3d(self.d_dim*4,  self.d_dim*8, 3, stride=1, padding=0, bias=True)
        self.gn_4 = nn.GroupNorm(8, self.d_dim*8)
        self.conv_style_discr = nn.Conv3d(self.d_dim*8, self.z_dim,    1, stride=1, padding=0, bias=True)
        self.gn_style_discr = nn.GroupNorm(1, self.z_dim)  # equivalent to Layer Normalization
        self.pool = nn.AdaptiveMaxPool3d(1)
        gn_init(self.gn_4)
        gn_init(self.gn_style_discr, zero_init=True)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.gn_4(self.conv_4(out))
        out = torch.relu(out)
        out = self.gn_style_discr(self.conv_style_discr(out))
        out = torch.relu(out)
        # out = F.normalize(out, p=2, dim=1)  # already use layer normalization
        return out

    def layer(self, voxels, ):
        out = voxels
        out4 = self.gn_4(self.conv_4(out))
        out4 = torch.relu(out4)
        out = self.gn_style_discr(self.conv_style_discr(out4))
        out = torch.relu(out)
        # out = F.normalize(out, p=2, dim=1)  # already use layer normalization
        out = torch.cat([self.pool(out).squeeze(2).squeeze(2).squeeze(2).T,
                         self.pool(out4).squeeze(2).squeeze(2).squeeze(2).T]).T
        return out


class discriminator_part_style_plausibility_4(nn.Module):
    def __init__(self, d_dim, z_dim, init_weights=False):
        super(discriminator_part_style_plausibility_4, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.conv_3 = nn.Conv3d(self.d_dim*2,  self.d_dim*4, 3, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.d_dim*4,  self.z_dim*4, 3, stride=1, padding=0, bias=True)
        self.conv_5 = nn.Conv3d(self.z_dim*4,  self.z_dim*2, 3, stride=1, padding=0, bias=True)
        self.conv_style_discr = nn.Conv3d(self.z_dim*2, self.z_dim,    1, stride=1, padding=0, bias=True)
        self.pool = nn.AdaptiveMaxPool3d(1)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.conv_3(out)
        out = torch.relu(out)
        out = self.conv_4(out)
        out = torch.relu(out)
        out = self.conv_5(out)
        out = torch.relu(out)
        out = self.conv_style_discr(out)
        out = torch.relu(out)
        out = F.normalize(out, p=2, dim=1)
        return out

    def layer(self, voxels, ):
        out = voxels
        out = self.conv_3(out)
        out = torch.relu(out)
        out4 = self.conv_4(out)
        out4 = torch.relu(out4)
        out5 = self.conv_5(out4)
        out5 = torch.relu(out5)
        out = self.conv_style_discr(out5)
        out = torch.relu(out)
        out = F.normalize(out, p=2, dim=1)
        out = torch.cat([self.pool(out).squeeze(2).squeeze(2).squeeze(2).T,
                         self.pool(out5).squeeze(2).squeeze(2).squeeze(2).T,
                         self.pool(out4).squeeze(2).squeeze(2).squeeze(2).T]).T
        return out


class discriminator_part_style_plausibility_4_gn(nn.Module):
    def __init__(self, d_dim, z_dim, init_weights=False):
        super(discriminator_part_style_plausibility_4_gn, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.conv_3 = nn.Conv3d(self.d_dim*2,  self.d_dim*4, 3, stride=1, padding=0, bias=True)
        self.gn_3 = nn.GroupNorm(4, self.d_dim*4)
        self.conv_4 = nn.Conv3d(self.d_dim*4,  self.z_dim*4, 3, stride=1, padding=0, bias=True)
        self.gn_4 = nn.GroupNorm(4, self.z_dim*4)
        self.conv_5 = nn.Conv3d(self.z_dim*4,  self.z_dim*2, 3, stride=1, padding=0, bias=True)
        self.gn_5 = nn.GroupNorm(2, self.z_dim*2)
        self.conv_style_discr = nn.Conv3d(self.z_dim*2, self.z_dim,    1, stride=1, padding=0, bias=True)
        self.gn_style_discr = nn.GroupNorm(1, self.z_dim)  # equivalent to Layer Normalization
        self.pool = nn.AdaptiveMaxPool3d(1)
        gn_init(self.gn_3)
        gn_init(self.gn_4)
        gn_init(self.gn_5)
        gn_init(self.gn_style_discr, zero_init=True)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.gn_3(self.conv_3(out))
        out = torch.relu(out)
        out = self.gn_4(self.conv_4(out))
        out = torch.relu(out)
        out = self.gn_5(self.conv_5(out))
        out = torch.relu(out)
        out = self.gn_style_discr(self.conv_style_discr(out))
        out = torch.relu(out)
        # out = F.normalize(out, p=2, dim=1)  # use layer normalization
        return out

    def layer(self, voxels, ):
        out = voxels
        out = self.gn_3(self.conv_3(out))
        out = torch.relu(out)
        out4 = self.gn_4(self.conv_4(out))
        out4 = torch.relu(out4)
        out5 = self.gn_5(self.conv_5(out4))
        out5 = torch.relu(out5)
        out = self.gn_style_discr(self.conv_style_discr(out5))
        out = torch.relu(out)
        # out = F.normalize(out, p=2, dim=1)  # use layer normalization
        out = torch.cat([self.pool(out).squeeze(2).squeeze(2).squeeze(2).T,
                         self.pool(out5).squeeze(2).squeeze(2).squeeze(2).T,
                         self.pool(out4).squeeze(2).squeeze(2).squeeze(2).T]).T
        return out


class discriminator_part_style_plausibility_2(nn.Module):
    def __init__(self, d_dim, z_dim, init_weights=False):
        super(discriminator_part_style_plausibility_2, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.conv_style_discr = nn.Conv3d(self.d_dim*16, self.z_dim,    1, stride=1, padding=0, bias=True)
        self.pool = nn.AdaptiveMaxPool3d(1)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            self.initialize_weights()

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.conv_style_discr(out)
        out = torch.sigmoid(out)
        return out

    def initialize_weights(self):
        nn.init.xavier_normal_(self.conv_style_discr.weight)
        nn.init.constant_(self.conv_style_discr.bias, 0)


class discriminator_part_style_plausibility_2_gn(nn.Module):
    def __init__(self, d_dim, z_dim, init_weights=False):
        super(discriminator_part_style_plausibility_2_gn, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.conv_style_discr = nn.Conv3d(self.d_dim*16, self.z_dim,    1, stride=1, padding=0, bias=True)
        self.gn_style_discr = nn.GroupNorm(1, self.z_dim)   # equivalent to Layer Normalization
        self.pool = nn.AdaptiveMaxPool3d(1)
        gn_init(self.gn_style_discr, zero_init=True)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            self.initialize_weights()

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.conv_style_discr(out)
        out = torch.sigmoid(out)
        return out

    def initialize_weights(self):
        nn.init.xavier_normal_(self.conv_style_discr.weight)
        nn.init.constant_(self.conv_style_discr.bias, 0)


class style_encoder_part_encode(nn.Module):
    def __init__(self, d_dim, z_dim, init_weights=False):
        super(style_encoder_part_encode, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.conv_style_enc = nn.Conv3d(self.d_dim*16, self.z_dim,    1, stride=1, padding=0, bias=True)
        self.pool = nn.AdaptiveMaxPool3d(1)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.conv_style_enc(out)
        out = torch.relu(out)
        out = F.normalize(out, p=2, dim=1)
        out = self.pool(out)
        return out


class style_encoder_8_with_norm(nn.Module):

    def __init__(self, norm_type='unit_length', init_weights=False):
        super(style_encoder_8_with_norm, self).__init__()

        self.conv_0 = nn.Conv3d(1,  4,    5, stride=1, dilation=1, padding=2, bias=True)
        self.conv_1 = nn.Conv3d(4,    8,  5, stride=1, dilation=2, padding=4, bias=True)
        self.pool = nn.AdaptiveMaxPool3d(1)
        self.norm_type = norm_type
        if self.norm_type == 'group_norm':
            self.gn_0 = torch.nn.GroupNorm(1, 4)
            self.gn_1 = torch.nn.GroupNorm(1, 8)
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)

    def forward(self, voxels, is_training=False):
        out = voxels

        out = self.conv_0(out)
        if self.norm_type == 'group_norm':
            out = self.gn_0(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_1(out)
        if self.norm_type == 'group_norm':
            out = self.gn_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.pool(out)
        if self.norm_type == 'unit_length':
            out = F.normalize(out, p=2, dim=1)

        return out

    def layer(self, voxels, layer='all'):
        out = voxels

        out0 = self.conv_0(out)
        if self.norm_type == 'group_norm':
            out0 = self.gn_0(out)
        out0 = F.leaky_relu(out0, negative_slope=0.02, inplace=True)

        out1 = self.conv_1(out0)
        if self.norm_type == 'group_norm':
            out1 = self.gn_1(out)
        out1 = F.leaky_relu(out1, negative_slope=0.02, inplace=True)
        if self.norm_type == 'unit_length':
            out1 = F.normalize(out1, p=2, dim=1)

        if layer == 'all':
            out = torch.cat([self.pool(out0).squeeze(2).squeeze(2).squeeze(2).T,
                             self.pool(out1).squeeze(2).squeeze(2).squeeze(2).T]).T
            return out


class style_encoder_generic(nn.Module):
    def __init__(self, pool_method='max', with_norm=False, norm_type='unit_length', init_weights=False):
        super(style_encoder_generic, self).__init__()
        self.convolutions = torch.nn.ModuleList()
        self.pool = nn.AdaptiveMaxPool3d(1)
        self.pool_method = pool_method
        self.with_norm = with_norm
        self.norm_type = norm_type
        if self.with_norm and (self.norm_type == 'group_norm'):
            self.gns = torch.nn.ModuleList()
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)

    def forward(self, voxels, is_training=False):
        out = voxels
        if self.with_norm and (self.norm_type == 'group_norm'):
            for conv, gn in zip(self.convolutions, self.gns):
                out = conv(out)
                out = gn(out)
                out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            for conv in self.convolutions:
                out = conv(out)
                out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        if self.pool_method == 'max':
            out = self.pool(out)
        else:
            out = torch.mean(out, dim=(2, 3, 4)).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        return out

    def layer(self, voxels, layer='all'):
        out = voxels

        out_keep = []

        if self.with_norm and (self.norm_type == 'group_norm'):
            for idx, (conv, gn) in enumerate(zip(self.convolutions, self.gns)):
                out = conv(out)
                out = gn(out)
                out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
                if idx >= len(self.convolutions) - 2:
                    out_keep.append(self.pool(out).squeeze(2).squeeze(2).squeeze(2).T)
        else:
            for idx, conv in enumerate(self.convolutions):
                out = conv(out)
                out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
                if idx >= len(self.convolutions) - 2:
                    out_keep.append(self.pool(out).squeeze(2).squeeze(2).squeeze(2).T)

        out = torch.cat(out_keep).T
        return out


class style_encoder_8_old(nn.Module):

    def __init__(self,  pool_method='max', kernel=5, dilation=True, with_norm=False, norm_type='unit_length', init_weights=None):
        super(style_encoder_8_old, self).__init__()

        self.conv_0 = nn.Conv3d(1,  4,    5, stride=1, dilation=1, padding=2, bias=True)
        self.conv_1 = nn.Conv3d(4,    8,  5, stride=1, dilation=2, padding=4, bias=True)
        self.pool = nn.AdaptiveMaxPool3d(1)
        self.with_norm = with_norm
        self.norm_type = norm_type
        if self.with_norm and (self.norm_type == 'group_norm'):
            self.gn_0 = torch.nn.GroupNorm(1, 4)
            self.gn_1 = torch.nn.GroupNorm(1, 8)

    def forward(self, voxels, is_training=False):
        out = voxels

        out = self.conv_0(out)
        if self.with_norm and (self.norm_type == 'group_norm'):
            out = self.gn_0(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_1(out)
        if self.with_norm and (self.norm_type == 'group_norm'):
            out = self.gn_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.pool(out)
        if self.with_norm and (self.norm_type == 'unit_length'):
            out = F.normalize(out, p=2, dim=1)

        return out

    def layer(self, voxels, layer='all'):
        out = voxels

        out0 = self.conv_0(out)
        if self.with_norm and (self.norm_type == 'group_norm'):
            out0 = self.gn_0(out0)
        out0 = F.leaky_relu(out0, negative_slope=0.02, inplace=True)

        out1 = self.conv_1(out0)
        if self.with_norm and (self.norm_type == 'group_norm'):
            out1 = self.gn_1(out1)
        out1 = F.leaky_relu(out1, negative_slope=0.02, inplace=True)

        if layer == 'all':
            out = torch.cat([self.pool(out0).squeeze(2).squeeze(2).squeeze(2).T,
                             self.pool(out1).squeeze(2).squeeze(2).squeeze(2).T]).T
            return out


class style_encoder_8(style_encoder_generic):
    def __init__(self, pool_method='max', kernel=5, dilation=True, with_norm=False, norm_type='unit_length', init_weights=False):
        super(style_encoder_8, self).__init__(pool_method, with_norm, norm_type)
        self.convolutions.append(
            nn.Conv3d(1, 4, kernel, stride=1, dilation=1, padding=2, bias=True)
        )
        self.convolutions.append(
            nn.Conv3d(4, 8, kernel, stride=1, dilation=2 if dilation else 1, padding=4 if dilation else 2, bias=True)
        )
        if self.with_norm and (self.norm_type == 'group_norm'):
            self.gns.append(torch.nn.GroupNorm(1, 4))
            self.gns.append(torch.nn.GroupNorm(1, 8))
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)


class style_encoder_16(style_encoder_generic):
    def __init__(self, pool_method='max', kernel=5, dilation=True, with_norm=False, norm_type='unit_length', init_weights=False):
        super(style_encoder_16, self).__init__(pool_method, with_norm, norm_type)
        self.convolutions.append(
            nn.Conv3d(1, 4, kernel, stride=1, dilation=1, padding=2, bias=True)
        )
        self.convolutions.append(
            nn.Conv3d(4, 8, kernel, stride=1, dilation=2 if dilation else 1, padding=4 if dilation else 2, bias=True)
        )
        self.convolutions.append(
            nn.Conv3d(8, 16, kernel, stride=1, dilation=2 if dilation else 1, padding=4 if dilation else 2, bias=True)
        )
        if self.with_norm and (self.norm_type == 'group_norm'):
            self.gns.append(torch.nn.GroupNorm(1, 4))
            self.gns.append(torch.nn.GroupNorm(1, 8))
            self.gns.append(torch.nn.GroupNorm(1, 16))
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)


class style_encoder_32(style_encoder_generic):
    def __init__(self, pool_method='max', kernel=5, dilation=True, with_norm=False, norm_type='unit_length', init_weights=False):
        super(style_encoder_32, self).__init__(pool_method, with_norm, norm_type)
        self.convolutions.append(
            nn.Conv3d(1, 4, kernel, stride=1, dilation=1, padding=2, bias=True)
        )
        self.convolutions.append(
            nn.Conv3d(4, 8, kernel, stride=1, dilation=2 if dilation else 1, padding=4 if dilation else 2, bias=True)
        )
        self.convolutions.append(
            nn.Conv3d(8, 16, kernel, stride=1, dilation=2 if dilation else 1, padding=4 if dilation else 2, bias=True)
        )
        self.convolutions.append(
            nn.Conv3d(16, 32, kernel, stride=1, dilation=1, padding=2, bias=True)
        )
        if self.with_norm and (self.norm_type == 'group_norm'):
            self.gns.append(torch.nn.GroupNorm(1, 4))
            self.gns.append(torch.nn.GroupNorm(1, 8))
            self.gns.append(torch.nn.GroupNorm(1, 16))
            self.gns.append(torch.nn.GroupNorm(1, 32))
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)


class style_encoder_64(style_encoder_generic):
    def __init__(self, pool_method='max', kernel=5, dilation=True, with_norm=False, norm_type='unit_length', init_weights=False):
        super(style_encoder_64, self).__init__(pool_method, with_norm, norm_type)
        self.convolutions.append(
            nn.Conv3d(1, 8, kernel, stride=1, dilation=1, padding=2, bias=True)
        )
        self.convolutions.append(
            nn.Conv3d(8, 16, kernel, stride=1, dilation=2 if dilation else 1, padding=4 if dilation else 2, bias=True)
        )
        self.convolutions.append(
            nn.Conv3d(16, 32, kernel, stride=1, dilation=2 if dilation else 1, padding=4 if dilation else 2, bias=True)
        )
        self.convolutions.append(
            nn.Conv3d(32, 64, kernel, stride=1, dilation=1, padding=2, bias=True)
        )
        if self.with_norm and (self.norm_type == 'group_norm'):
            self.gns.append(torch.nn.GroupNorm(1, 8))
            self.gns.append(torch.nn.GroupNorm(1, 16))
            self.gns.append(torch.nn.GroupNorm(1, 32))
            self.gns.append(torch.nn.GroupNorm(1, 64))
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)


class style_encoder_128_old(style_encoder_generic):
    def __init__(self, pool_method='max', kernel=5, dilation=True):
        super(style_encoder_128_old, self).__init__(pool_method)
        self.convolutions.append(
            nn.Conv3d(1, 4, 5, stride=1, dilation=1, padding=2, bias=True)
        )
        self.convolutions.append(
            nn.Conv3d(4, 8, 4, stride=1, dilation=2, padding=4, bias=True)
        )
        self.convolutions.append(
            nn.Conv3d(8, 16, 3, stride=1, dilation=2, padding=4, bias=True)
        )
        self.convolutions.append(
            nn.Conv3d(16, 32, 3, stride=1, dilation=2, padding=4, bias=True)
        )
        self.convolutions.append(
            nn.Conv3d(32, 64, 3, stride=1, dilation=1, padding=2, bias=True)
        )
        self.convolutions.append(
            nn.Conv3d(64, 128, 1, stride=1, dilation=1, padding=0, bias=True)
        )


class style_encoder_128(style_encoder_generic):
    def __init__(self, pool_method='max', kernel=5, dilation=True, with_norm=False, norm_type='unit_length', init_weights=False):
        super(style_encoder_128, self).__init__(pool_method, with_norm, norm_type)
        self.convolutions.append(
            nn.Conv3d(1, 8, kernel, stride=1, dilation=1, padding=2, bias=True)
        )
        self.convolutions.append(
            nn.Conv3d(8, 16, kernel, stride=1, dilation=2 if dilation else 1, padding=4 if dilation else 2, bias=True)
        )
        self.convolutions.append(
            nn.Conv3d(16, 32, kernel, stride=1, dilation=2 if dilation else 1, padding=4 if dilation else 2, bias=True)
        )
        self.convolutions.append(
            nn.Conv3d(32, 64, kernel, stride=1, dilation=2 if dilation else 1, padding=4 if dilation else 2, bias=True)
        )
        self.convolutions.append(
            nn.Conv3d(64, 128, kernel, stride=1, dilation=2 if dilation else 1, padding=4 if dilation else 2, bias=True)
        )
        if self.with_norm and (self.norm_type == 'group_norm'):
            self.gns.append(torch.nn.GroupNorm(1, 8))
            self.gns.append(torch.nn.GroupNorm(1, 16))
            self.gns.append(torch.nn.GroupNorm(1, 32))
            self.gns.append(torch.nn.GroupNorm(1, 64))
            self.gns.append(torch.nn.GroupNorm(1, 128))
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            initialize_conv_relu_layers(self)


#64 -> 256
class generator(nn.Module):

    def _init_config(self, g_dim, prob_dim, z_dim, sigmoid=False, leaky=True):
        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim
        self.sigmoid = sigmoid
        self.leaky = True

    def _init_style_embedding(self):
        style_codes = torch.zeros((self.prob_dim, self.z_dim))
        self.style_codes = nn.Parameter(style_codes)
        nn.init.constant_(self.style_codes, 0.0)

    def _init_net(self):
        self.conv_0 = nn.Conv3d(1+self.z_dim,             self.g_dim,    5, stride=1, dilation=1, padding=2, bias=True)
        self.conv_1 = nn.Conv3d(self.g_dim+self.z_dim,    self.g_dim*2,  5, stride=1, dilation=2, padding=4, bias=True)
        self.conv_2 = nn.Conv3d(self.g_dim*2+self.z_dim,  self.g_dim*4,  5, stride=1, dilation=2, padding=4, bias=True)
        self.conv_3 = nn.Conv3d(self.g_dim*4+self.z_dim,  self.g_dim*8,  5, stride=1, dilation=1, padding=2, bias=True)
        self.conv_4 = nn.Conv3d(self.g_dim*8+self.z_dim,  self.g_dim*4,  5, stride=1, dilation=1, padding=2, bias=True)

        self.conv_5 = nn.ConvTranspose3d(self.g_dim*4,  self.g_dim*2, 4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim*2,  self.g_dim*2,  3, stride=1, padding=1, bias=True)
        self.conv_7 = nn.ConvTranspose3d(self.g_dim*2,  self.g_dim,   4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim,    1,             3, stride=1, padding=1, bias=True)

    def __init__(self, g_dim, prob_dim, z_dim, sigmoid=False, leaky=True, init_weights=False):
        super(generator, self).__init__()
        self._init_config(g_dim, prob_dim, z_dim, sigmoid, leaky)
        self._init_style_embedding()
        self._init_net()
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            self.initialize_weights()

    def forward(self, voxels, z, mask_, is_training=False):
        out = voxels
        mask = F.interpolate(mask_, scale_factor=4, mode='nearest')

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_0(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_1(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_2(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_3(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_4(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_5(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_6(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_7(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_8(out)
        #out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        #out = out.clamp(max=1.0)
        if self.sigmoid:
            out = torch.sigmoid(out)
        else:
            out = torch.max(torch.min(out, out * 0.002 + 0.998), out * 0.002)
        #out = torch.sigmoid(out)

        out = out*mask

        return out

    def initialize_weights(self):
        for conv in [self.conv_0, self.conv_1, self.conv_2, self.conv_3, self.conv_4,
                     self.conv_5, self.conv_6, self.conv_7]:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.constant_(conv.bias, 0)
        nn.init.xavier_normal_(self.conv_8.weight)
        nn.init.constant_(self.conv_8.bias, 0)


#64 -> 256
class generator_gn(generator):

    def _init_net(self):
        self.conv_0 = nn.Conv3d(1+self.z_dim,             self.g_dim,    5, stride=1, dilation=1, padding=2, bias=True)
        self.gn_0 = nn.GroupNorm(1, self.g_dim)
        self.conv_1 = nn.Conv3d(self.g_dim+self.z_dim,    self.g_dim*2,  5, stride=1, dilation=2, padding=4, bias=True)
        self.gn_1 = nn.GroupNorm(2, self.g_dim*2)
        self.conv_2 = nn.Conv3d(self.g_dim*2+self.z_dim,  self.g_dim*4,  5, stride=1, dilation=2, padding=4, bias=True)
        self.gn_2 = nn.GroupNorm(4, self.g_dim*4)
        self.conv_3 = nn.Conv3d(self.g_dim*4+self.z_dim,  self.g_dim*8,  5, stride=1, dilation=1, padding=2, bias=True)
        self.gn_3 = nn.GroupNorm(8, self.g_dim*8)
        self.conv_4 = nn.Conv3d(self.g_dim*8+self.z_dim,  self.g_dim*4,  5, stride=1, dilation=1, padding=2, bias=True)
        self.gn_4 = nn.GroupNorm(4, self.g_dim*4)

        self.conv_5 = nn.ConvTranspose3d(self.g_dim*4,  self.g_dim*2, 4, stride=2, padding=1, bias=True)
        self.gn_5 = nn.GroupNorm(2, self.g_dim*2)
        self.conv_6 = nn.Conv3d(self.g_dim*2,  self.g_dim*2,  3, stride=1, padding=1, bias=True)
        self.gn_6 = nn.GroupNorm(2, self.g_dim*2)
        self.conv_7 = nn.ConvTranspose3d(self.g_dim*2,  self.g_dim,   4, stride=2, padding=1, bias=True)
        self.gn_7 = nn.GroupNorm(1, self.g_dim)
        self.conv_8 = nn.Conv3d(self.g_dim,    1,             3, stride=1, padding=1, bias=True)

        gn_init(self.gn_0)
        gn_init(self.gn_1)
        gn_init(self.gn_2)
        gn_init(self.gn_3)
        gn_init(self.gn_4, zero_init=True)
        gn_init(self.gn_5)
        gn_init(self.gn_6)
        gn_init(self.gn_7)
        gn_init(self.gn_8, zero_init=True)

    def __init__(self, g_dim, prob_dim, z_dim, sigmoid=False, leaky=True, init_weights=False):
        super(generator_gn, self).__init__(g_dim, prob_dim, z_dim, sigmoid, leaky, init_weights)

    def forward(self, voxels, z, mask_, is_training=False):
        out = voxels
        mask = F.interpolate(mask_, scale_factor=4, mode='nearest')

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_0(self.conv_0(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_1(self.conv_1(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_2(self.conv_2(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_3(self.conv_3(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_4(self.conv_4(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.gn_5(self.conv_5(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.gn_6(self.conv_6(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.gn_7(self.conv_7(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.gn_8(self.conv_8(out))
        #out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        #out = out.clamp(max=1.0)
        if self.sigmoid:
            out = torch.sigmoid(out)
        else:
            out = torch.max(torch.min(out, out * 0.002 + 0.998), out * 0.002)
        #out = torch.sigmoid(out)

        out = out*mask

        return out


# 64 -> 256
class generator_allstyles(generator):

    def _init_config(self, g_dim, prob_dim, z_dim, sigmoid=False, leaky=True):
        self.g_dim = g_dim
        self.z_dim = z_dim
        self.sigmoid = sigmoid
        self.leaky = leaky

    def _init_style_embedding(self):
        pass

    def __init__(self, g_dim, z_dim, sigmoid=False, leaky=True, init_weights=False):
        super(generator_allstyles, self).__init__(g_dim, None, z_dim, sigmoid, leaky, init_weights)


# 64 -> 256
class generator_allstyles_gn(generator_gn):

    def _init_config(self, g_dim, prob_dim, z_dim, sigmoid=False, leaky=True):
        self.g_dim = g_dim
        self.z_dim = z_dim
        self.sigmoid = sigmoid
        self.leaky = leaky

    def _init_style_embedding(self):
        pass

    def __init__(self, g_dim, z_dim, sigmoid=False, leaky=True, init_weights=False):
        super(generator_allstyles_gn, self).__init__(g_dim, None, z_dim, sigmoid, leaky, init_weights)


#32 -> 128
class generator_halfsize(nn.Module):

    def _init_config(self, g_dim, prob_dim, z_dim, sigmoid=False, leaky=True):
        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim
        self.sigmoid = sigmoid
        self.leaky = leaky

    def _init_style_embedding(self):
        style_codes = torch.zeros((self.prob_dim, self.z_dim))
        self.style_codes = nn.Parameter(style_codes)
        nn.init.constant_(self.style_codes, 0.0)

    def _init_net(self):
        self.conv_0 = nn.Conv3d(1+self.z_dim,             self.g_dim,    3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_1 = nn.Conv3d(self.g_dim+self.z_dim,    self.g_dim*2,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_2 = nn.Conv3d(self.g_dim*2+self.z_dim,  self.g_dim*4,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_3 = nn.Conv3d(self.g_dim*4+self.z_dim,  self.g_dim*8,  3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.g_dim*8+self.z_dim,  self.g_dim*4,  3, stride=1, dilation=1, padding=1, bias=True)

        self.conv_5 = nn.ConvTranspose3d(self.g_dim*4,  self.g_dim*2, 4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim*2,  self.g_dim*2,  3, stride=1, padding=1, bias=True)
        self.conv_7 = nn.ConvTranspose3d(self.g_dim*2,  self.g_dim,   4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim,    1,             3, stride=1, padding=1, bias=True)

    def __init__(self, g_dim, prob_dim, z_dim, sigmoid=False, leaky=True, init_weights=False):
        super(generator_halfsize, self).__init__()
        self._init_config(g_dim, prob_dim, z_dim, sigmoid, leaky)
        self._init_style_embedding()
        self._init_net()
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            self.initialize_weights()

    def forward(self, voxels, z, mask_, is_training=False):
        out = voxels
        mask = F.interpolate(mask_, scale_factor=4, mode='nearest')

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_0(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_1(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_2(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_3(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_4(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_5(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_6(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_7(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_8(out)
        #out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        #out = out.clamp(max=1.0)
        if self.sigmoid:
            out = torch.sigmoid(out)
        else:
            out = torch.max(torch.min(out, out*0.002+0.998), out*0.002)
        #out = torch.sigmoid(out)

        out = out*mask

        return out

    def initialize_weights(self):
        for conv in [self.conv_0, self.conv_1, self.conv_2, self.conv_3, self.conv_4,
                     self.conv_5, self.conv_6, self.conv_7]:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.constant_(conv.bias, 0)
        nn.init.xavier_normal_(self.conv_8.weight)
        nn.init.constant_(self.conv_8.bias, 0)


#32 -> 128
class generator_halfsize_gn(generator_halfsize):

    def _init_net(self):
        self.conv_0 = nn.Conv3d(1+self.z_dim,             self.g_dim,    3, stride=1, dilation=1, padding=1, bias=True)
        self.gn_0 = nn.GroupNorm(1, self.g_dim)
        self.conv_1 = nn.Conv3d(self.g_dim+self.z_dim,    self.g_dim*2,  3, stride=1, dilation=2, padding=2, bias=True)
        self.gn_1 = nn.GroupNorm(2, self.g_dim*2)
        self.conv_2 = nn.Conv3d(self.g_dim*2+self.z_dim,  self.g_dim*4,  3, stride=1, dilation=2, padding=2, bias=True)
        self.gn_2 = nn.GroupNorm(4, self.g_dim*4)
        self.conv_3 = nn.Conv3d(self.g_dim*4+self.z_dim,  self.g_dim*8,  3, stride=1, dilation=1, padding=1, bias=True)
        self.gn_3 = nn.GroupNorm(8, self.g_dim*8)
        self.conv_4 = nn.Conv3d(self.g_dim*8+self.z_dim,  self.g_dim*4,  3, stride=1, dilation=1, padding=1, bias=True)
        self.gn_4 = nn.GroupNorm(4, self.g_dim*4)

        self.conv_5 = nn.ConvTranspose3d(self.g_dim*4,  self.g_dim*2, 4, stride=2, padding=1, bias=True)
        self.gn_5 = nn.GroupNorm(2, self.g_dim*2)
        self.conv_6 = nn.Conv3d(self.g_dim*2,  self.g_dim*2,  3, stride=1, padding=1, bias=True)
        self.gn_6 = nn.GroupNorm(2, self.g_dim*2)
        self.conv_7 = nn.ConvTranspose3d(self.g_dim*2,  self.g_dim,   4, stride=2, padding=1, bias=True)
        self.gn_7 = nn.GroupNorm(1, self.g_dim)
        self.conv_8 = nn.Conv3d(self.g_dim,    1,             3, stride=1, padding=1, bias=True)

        gn_init(self.gn_0)
        gn_init(self.gn_1)
        gn_init(self.gn_2)
        gn_init(self.gn_3)
        gn_init(self.gn_4, zero_init=True)
        gn_init(self.gn_5)
        gn_init(self.gn_6)
        gn_init(self.gn_7)
        gn_init(self.gn_8, zero_init=True)

    def __init__(self, g_dim, prob_dim, z_dim, sigmoid=False, leaky=True, init_weights=False):
        super(generator_halfsize_gn, self).__init__(g_dim, prob_dim, z_dim, sigmoid, leaky, init_weights)

    def forward(self, voxels, z, mask_, is_training=False):
        out = voxels
        mask = F.interpolate(mask_, scale_factor=4, mode='nearest')

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_0(self.conv_0(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_1(self.conv_1(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_2(self.conv_2(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_3(self.conv_3(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_4(self.conv_4(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.gn_5(self.conv_5(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.gn_6(self.conv_6(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.gn_7(self.conv_7(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.gn_8(self.conv_8(out))
        #out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        #out = out.clamp(max=1.0)
        if self.sigmoid:
            out = torch.sigmoid(out)
        else:
            out = torch.max(torch.min(out, out*0.002+0.998), out*0.002)
        #out = torch.sigmoid(out)

        out = out*mask

        return out


# 32 -> 128
class generator_halfsize_allstyles(generator_halfsize):

    def _init_config(self, g_dim, prob_dim, z_dim, sigmoid=False, leaky=True):
        self.g_dim = g_dim
        self.z_dim = z_dim
        self.sigmoid = sigmoid
        self.leaky = leaky

    def _init_style_embedding(self):
        pass

    def __init__(self, g_dim, z_dim, sigmoid=False, leaky=True, init_weights=False):
        super(generator_halfsize_allstyles, self).__init__(g_dim, None, z_dim, sigmoid, leaky, init_weights)


# 32 -> 128
class generator_halfsize_allstyles_gn(generator_halfsize_gn):

    def _init_config(self, g_dim, prob_dim, z_dim, sigmoid=False, leaky=True):
        self.g_dim = g_dim
        self.z_dim = z_dim
        self.sigmoid = sigmoid
        self.leaky = leaky

    def _init_style_embedding(self):
        pass

    def __init__(self, g_dim, z_dim, sigmoid=False, leaky=True, init_weights=False):
        super(generator_halfsize_allstyles_gn, self).__init__(g_dim, None, z_dim, sigmoid, leaky, init_weights)


#32 -> 256
class generator_halfsize_x8(nn.Module):

    def _init_config(self, g_dim, prob_dim, z_dim, sigmoid=False, leaky=True):
        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim
        self.sigmoid = sigmoid
        self.leaky = leaky

    def _init_style_embedding(self):
        style_codes = torch.zeros((self.prob_dim, self.z_dim))
        self.style_codes = nn.Parameter(style_codes)
        nn.init.constant_(self.style_codes, 0.0)

    def _init_net(self):

        self.conv_0 = nn.Conv3d(1+self.z_dim,             self.g_dim,    3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_1 = nn.Conv3d(self.g_dim+self.z_dim,    self.g_dim*2,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_2 = nn.Conv3d(self.g_dim*2+self.z_dim,  self.g_dim*4,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_3 = nn.Conv3d(self.g_dim*4+self.z_dim,  self.g_dim*8,  3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.g_dim*8+self.z_dim,  self.g_dim*8,  3, stride=1, dilation=1, padding=1, bias=True)

        self.conv_5 = nn.ConvTranspose3d(self.g_dim*8,  self.g_dim*4, 4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim*4,  self.g_dim*4,  3, stride=1, padding=1, bias=True)
        self.conv_7 = nn.ConvTranspose3d(self.g_dim*4,  self.g_dim*2, 4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim*2,  self.g_dim*2,  3, stride=1, padding=1, bias=True)
        self.conv_9 = nn.ConvTranspose3d(self.g_dim*2,  self.g_dim,   4, stride=2, padding=1, bias=True)
        self.conv_10 = nn.Conv3d(self.g_dim,   1,             3, stride=1, padding=1, bias=True)

    def __init__(self, g_dim, prob_dim, z_dim, sigmoid=False, leaky=True, init_weights=False):
        super(generator_halfsize_x8, self).__init__()
        self._init_config(g_dim, prob_dim, z_dim, sigmoid, leaky)
        self._init_style_embedding()
        self._init_net()
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            self.initialize_weights()

    def forward(self, voxels, z, mask_, is_training=False):
        out = voxels
        mask = F.interpolate(mask_, scale_factor=4, mode='nearest')

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_0(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_1(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_2(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_3(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_4(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_5(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_6(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_7(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_8(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_9(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_10(out)
        #out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        #out = out.clamp(max=1.0)
        if self.sigmoid:
            out = torch.sigmoid(out)
        else:
            out = torch.max(torch.min(out, out*0.002+0.998), out*0.002)
        #out = torch.sigmoid(out)

        out = out*mask

        return out

    def initialize_weights(self):
        for conv in [self.conv_0, self.conv_1, self.conv_2, self.conv_3, self.conv_4,
                     self.conv_5, self.conv_6, self.conv_7, self.conv_8, self.conv_9]:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.constant_(conv.bias, 0)
        nn.init.xavier_normal_(self.conv_10.weight)
        nn.init.constant_(self.conv_10.bias, 0)


#32 -> 256
class generator_halfsize_x8_gn(generator_halfsize_x8):

    def _init_net(self):

        self.conv_0 = nn.Conv3d(1+self.z_dim,             self.g_dim,    3, stride=1, dilation=1, padding=1, bias=True)
        self.gn_0 = nn.GroupNorm(1, self.g_dim)
        self.conv_1 = nn.Conv3d(self.g_dim+self.z_dim,    self.g_dim*2,  3, stride=1, dilation=2, padding=2, bias=True)
        self.gn_1 = nn.GroupNorm(2, self.g_dim*2)
        self.conv_2 = nn.Conv3d(self.g_dim*2+self.z_dim,  self.g_dim*4,  3, stride=1, dilation=2, padding=2, bias=True)
        self.gn_2 = nn.GroupNorm(4, self.g_dim*4)
        self.conv_3 = nn.Conv3d(self.g_dim*4+self.z_dim,  self.g_dim*8,  3, stride=1, dilation=1, padding=1, bias=True)
        self.gn_3 = nn.GroupNorm(8, self.g_dim*8)
        self.conv_4 = nn.Conv3d(self.g_dim*8+self.z_dim,  self.g_dim*8,  3, stride=1, dilation=1, padding=1, bias=True)
        self.gn_4 = nn.GroupNorm(8, self.g_dim*8)

        self.conv_5 = nn.ConvTranspose3d(self.g_dim*8,  self.g_dim*4, 4, stride=2, padding=1, bias=True)
        self.gn_5 = nn.GroupNorm(4, self.g_dim*4)
        self.conv_6 = nn.Conv3d(self.g_dim*4,  self.g_dim*4,  3, stride=1, padding=1, bias=True)
        self.gn_6 = nn.GroupNorm(4, self.g_dim*4)
        self.conv_7 = nn.ConvTranspose3d(self.g_dim*4,  self.g_dim*2, 4, stride=2, padding=1, bias=True)
        self.gn_7 = nn.GroupNorm(2, self.g_dim*2)
        self.conv_8 = nn.Conv3d(self.g_dim*2,  self.g_dim*2,  3, stride=1, padding=1, bias=True)
        self.gn_8 = nn.GroupNorm(2, self.g_dim*2)
        self.conv_9 = nn.ConvTranspose3d(self.g_dim*2,  self.g_dim,   4, stride=2, padding=1, bias=True)
        self.gn_9 = nn.GroupNorm(1, self.g_dim)
        self.conv_10 = nn.Conv3d(self.g_dim,   1,             3, stride=1, padding=1, bias=True)

        gn_init(self.gn_0)
        gn_init(self.gn_1)
        gn_init(self.gn_2)
        gn_init(self.gn_3)
        gn_init(self.gn_4, zero_init=True)
        gn_init(self.gn_5)
        gn_init(self.gn_6)
        gn_init(self.gn_7)
        gn_init(self.gn_8)
        gn_init(self.gn_9, zero_init=True)

    def __init__(self, g_dim, prob_dim, z_dim, sigmoid=False, leaky=True, init_weights=False):
        super(generator_halfsize_x8_gn, self).__init__(g_dim, prob_dim, z_dim, sigmoid, leaky, init_weights)
        self._init_config(g_dim, prob_dim, z_dim, sigmoid, leaky)
        self._init_style_embedding()
        self._init_net()

    def forward(self, voxels, z, mask_, is_training=False):
        out = voxels
        mask = F.interpolate(mask_, scale_factor=4, mode='nearest')

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_0(self.conv_0(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_1(self.conv_1(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_2(self.conv_2(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_3(self.conv_3(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_4(self.conv_4(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.gn_5(self.conv_5(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.gn_6(self.conv_6(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.gn_7(self.conv_7(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.gn_8(self.conv_8(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.gn_9(self.conv_9(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_10(out)
        #out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        #out = out.clamp(max=1.0)
        if self.sigmoid:
            out = torch.sigmoid(out)
        else:
            out = torch.max(torch.min(out, out*0.002+0.998), out*0.002)
        #out = torch.sigmoid(out)

        out = out*mask

        return out


#32 -> 256
class generator_halfsize_x8_allstyles(generator_halfsize_x8):

    def _init_config(self, g_dim, prob_dim, z_dim, sigmoid=False, leaky=True):
        self.g_dim = g_dim
        self.z_dim = z_dim
        self.sigmoid = sigmoid
        self.leaky = leaky

    def _init_style_embedding(self):
        pass

    def __init__(self, g_dim, z_dim, sigmoid=False, leaky=True, init_weights=False):
        super(generator_halfsize_x8_allstyles, self).__init__(g_dim, None, z_dim, sigmoid, leaky, init_weights)

    def forward_without_mask(self, voxels, z, is_training=False):
        out = voxels

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_0(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_1(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_2(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_3(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_4(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_5(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_6(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_7(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_8(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_9(out)
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_10(out)
        #out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        #out = out.clamp(max=1.0)
        out = torch.max(torch.min(out, out*0.002+0.998), out*0.002)
        #out = torch.sigmoid(out)

        return out


#32 -> 256
class generator_halfsize_x8_allstyles_gn(generator_halfsize_x8_gn):

    def _init_config(self, g_dim, prob_dim, z_dim, sigmoid=False, leaky=True):
        self.g_dim = g_dim
        self.z_dim = z_dim
        self.sigmoid = sigmoid
        self.leaky = leaky

    def _init_style_embedding(self):
        pass

    def __init__(self, g_dim, z_dim, sigmoid=False, leaky=True, init_weights=False):
        super(generator_halfsize_x8_allstyles_gn, self).__init__(g_dim, None, z_dim, sigmoid, leaky, init_weights)

    def forward_without_mask(self, voxels, z, is_training=False):
        out = voxels

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_0(self.conv_0(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_1(self.conv_1(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_2(self.conv_2(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_3(self.conv_3(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.gn_4(self.conv_4(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.gn_5(self.conv_5(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.gn_6(self.conv_6(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.gn_7(self.conv_7(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.gn_8(self.conv_8(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.gn_9(self.conv_9(out))
        if self.leaky:
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        else:
            out = torch.relu(out)

        out = self.conv_10(out)
        #out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        #out = out.clamp(max=1.0)
        out = torch.max(torch.min(out, out*0.002+0.998), out*0.002)
        #out = torch.sigmoid(out)

        return out


class adain_function(object):

    @staticmethod
    def calc_mean_std(feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 5)
        N, C = size[:2]  # batch, channels
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1, 1)
        return feat_mean, feat_std  # mean and std per sample,channel

    @staticmethod
    def adaptive_instance_normalization(content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = adain_function.calc_mean_std(style_feat)
        content_mean, content_std = adain_function.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class generator_adain_generic(nn.Module):

    def __init__(self, g_dim, sigmoid=False, alpha=1., init_weights=False):
        super(generator_adain_generic, self).__init__()
        self._init_config(g_dim, sigmoid, alpha)
        self._init_style_embedding()
        self._init_net()
        if init_weights:
            print(f"Initializing {self.__class__.__name__} weights")
            self.initialize_weights()

    def _init_config(self, g_dim, sigmoid=False, alpha=1.):
        self.g_dim = g_dim
        self.sigmoid = sigmoid
        self.alpha = alpha

    def _init_style_embedding(self):
        pass


class generator_halfsize_x8_adain(generator_adain_generic):

    def _init_net(self):
        self.conv_0 = nn.Conv3d(1,             self.g_dim,    3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_1 = nn.Conv3d(self.g_dim,    self.g_dim*2,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_2 = nn.Conv3d(self.g_dim*2,  self.g_dim*4,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_3 = nn.Conv3d(self.g_dim*4,  self.g_dim*8,  3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.g_dim*8,  self.g_dim*8,  3, stride=1, dilation=1, padding=1, bias=True)

        self.conv_5 = nn.ConvTranspose3d(self.g_dim*8,  self.g_dim*4,   4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim*4,           self.g_dim*4,   3, stride=1, padding=1, bias=True)
        self.conv_7 = nn.ConvTranspose3d(self.g_dim*4,  self.g_dim*2,   4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim*2,           self.g_dim*2,   3, stride=1, padding=1, bias=True)
        self.conv_9 = nn.ConvTranspose3d(self.g_dim*2,  self.g_dim,     4, stride=2, padding=1, bias=True)
        self.conv_10 = nn.Conv3d(self.g_dim,            1,              3, stride=1, padding=1, bias=True)

        self.pool = nn.AdaptiveMaxPool3d(1)

    def __init__(self, g_dim, sigmoid=False, alpha=1., init_weights=False):
        super(generator_halfsize_x8_adain, self).__init__(g_dim, sigmoid, alpha, init_weights)

    def forward(self, voxels_content, voxels_style, mask_, is_training=False):
        mask = F.interpolate(mask_, scale_factor=4, mode='nearest')

        out_content = self.conv_0(voxels_content)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)
        out_content = self.conv_1(out_content)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)
        out_content = self.conv_2(out_content)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)
        out_content = self.conv_3(out_content)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)
        out_content = self.conv_4(out_content)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)

        out_style = self.conv_0(voxels_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        out_style = self.conv_1(out_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        out_style = self.conv_2(out_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        out_style = self.conv_3(out_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        out_style = self.conv_4(out_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)

        t = adain_function.adaptive_instance_normalization(out_content, out_style)
        t = self.alpha * t + (1 - self.alpha) * out_content

        out = self.conv_5(t)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_8(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_9(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_10(out)
        #out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        #out = out.clamp(max=1.0)
        if self.sigmoid:
            out = torch.sigmoid(out)
        else:
            out = torch.max(torch.min(out, out*0.002+0.998), out*0.002)
        #out = torch.sigmoid(out)

        out = out*mask

        return out

    def layer(self, voxels_style, layer='all'):

        out_style = self.conv_0(voxels_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        out_style = self.conv_1(out_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        out_style2 = self.conv_2(out_style)
        out_style2 = F.leaky_relu(out_style2, negative_slope=0.02, inplace=True)
        out_style3 = self.conv_3(out_style2)
        out_style3 = F.leaky_relu(out_style3, negative_slope=0.02, inplace=True)
        out_style4 = self.conv_4(out_style3)
        out_style4 = F.leaky_relu(out_style4, negative_slope=0.02, inplace=True)

        if layer == 'all':
            out = torch.cat([self.pool(out_style4).squeeze(2).squeeze(2).squeeze(2).T,
                             self.pool(out_style3).squeeze(2).squeeze(2).squeeze(2).T,
                             self.pool(out_style2).squeeze(2).squeeze(2).squeeze(2).T]).T
            return out

        return self.pool(out_style4).squeeze(2).squeeze(2).squeeze(2)

    def initialize_weights(self):
        for conv in [self.conv_0, self.conv_1, self.conv_2, self.conv_3, self.conv_4,
                     self.conv_5, self.conv_6, self.conv_7, self.conv_8, self.conv_9]:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.constant_(conv.bias, 0)
        nn.init.xavier_normal_(self.conv_10.weight)
        nn.init.constant_(self.conv_10.bias, 0)


class generator_halfsize_x8_adain_layers(generator_halfsize_x8_adain):

    def __init__(self, g_dim, sigmoid=False, alpha=0.5, init_weights=False):
        super(generator_halfsize_x8_adain_layers, self).__init__(g_dim, sigmoid, alpha, init_weights)

    def forward(self, voxels_content, voxels_style, mask_, is_training=False):
        mask = F.interpolate(mask_, scale_factor=4, mode='nearest')

        out_content = self.conv_0(voxels_content)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)
        out_style = self.conv_0(voxels_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        t = adain_function.adaptive_instance_normalization(out_content, out_style)
        t = self.alpha * t + (1 - self.alpha) * out_content

        out_content = self.conv_1(t)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)
        out_style = self.conv_1(out_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        t = adain_function.adaptive_instance_normalization(out_content, out_style)
        t = self.alpha * t + (1 - self.alpha) * out_content

        out_content = self.conv_2(t)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)
        out_style = self.conv_2(out_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        t = adain_function.adaptive_instance_normalization(out_content, out_style)
        t = self.alpha * t + (1 - self.alpha) * out_content

        out_content = self.conv_3(t)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)
        out_style = self.conv_3(out_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        t = adain_function.adaptive_instance_normalization(out_content, out_style)
        t = self.alpha * t + (1 - self.alpha) * out_content

        out_content = self.conv_4(t)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)
        out_style = self.conv_4(out_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        t = adain_function.adaptive_instance_normalization(out_content, out_style)
        t = self.alpha * t + (1 - self.alpha) * out_content

        out = self.conv_5(t)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_8(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_9(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_10(out)
        #out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        #out = out.clamp(max=1.0)
        if self.sigmoid:
            out = torch.sigmoid(out)
        else:
            out = torch.max(torch.min(out, out*0.002+0.998), out*0.002)
        #out = torch.sigmoid(out)

        out = out*mask

        return out

    def layer(self, voxels_style, layer='all'):
        raise Exception('unimplemented')


class generator_halfsize_x8_adain_share(generator_halfsize_x8_adain):

    def set_shared_weights(self, discriminator):
        assert self.g_dim == discriminator.d_dim
        self.conv_1.weight.data = discriminator.conv_2.weight.data
        self.conv_1.bias.data = discriminator.conv_2.bias.data
        self.conv_2.weight.data = discriminator.conv_3.weight.data
        self.conv_2.bias.data = discriminator.conv_3.bias.data
        self.conv_2.weight.data = discriminator.conv_3.weight.data
        self.conv_2.bias.data = discriminator.conv_3.bias.data


class generator_halfsize_adain(generator_adain_generic):

    def _init_net(self):
        self.conv_0 = nn.Conv3d(1,             self.g_dim,    3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_1 = nn.Conv3d(self.g_dim,    self.g_dim*2,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_2 = nn.Conv3d(self.g_dim*2,  self.g_dim*4,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_3 = nn.Conv3d(self.g_dim*4,  self.g_dim*8,  3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.g_dim*8,  self.g_dim*4,  3, stride=1, dilation=1, padding=1, bias=True)

        self.conv_5 = nn.ConvTranspose3d(self.g_dim*4,  self.g_dim*2,   4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim*2,  self.g_dim*2,  3, stride=1, padding=1, bias=True)
        self.conv_7 = nn.ConvTranspose3d(self.g_dim*2,  self.g_dim,   4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim,    1,             3, stride=1, padding=1, bias=True)

    def __init__(self, g_dim, sigmoid=False, alpha=1., init_weights=False):
        super(generator_halfsize_adain, self).__init__(g_dim, sigmoid, alpha, init_weights)

    def forward(self, voxels_content, voxels_style, mask_, is_training=False):
        mask = F.interpolate(mask_, scale_factor=4, mode='nearest')

        out_content = self.conv_0(voxels_content)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)
        out_content = self.conv_1(out_content)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)
        out_content = self.conv_2(out_content)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)
        out_content = self.conv_3(out_content)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)
        out_content = self.conv_4(out_content)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)

        out_style = self.conv_0(voxels_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        out_style = self.conv_1(out_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        out_style = self.conv_2(out_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        out_style = self.conv_3(out_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        out_style = self.conv_4(out_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)

        t = adain_function.adaptive_instance_normalization(out_content, out_style)
        t = self.alpha * t + (1 - self.alpha) * out_content

        out = self.conv_5(t)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_8(out)

        #out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        #out = out.clamp(max=1.0)
        if self.sigmoid:
            out = torch.sigmoid(out)
        else:
            out = torch.max(torch.min(out, out*0.002+0.998), out*0.002)
        #out = torch.sigmoid(out)

        out = out*mask

        return out

    def initialize_weights(self):
        for conv in [self.conv_0, self.conv_1, self.conv_2, self.conv_3, self.conv_4,
                     self.conv_5, self.conv_6, self.conv_7]:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.constant_(conv.bias, 0)
        nn.init.xavier_normal_(self.conv_8.weight)
        nn.init.constant_(self.conv_8.bias, 0)


class generator_adain(generator_adain_generic):

    def _init_net(self):
        self.conv_0 = nn.Conv3d(1,             self.g_dim,    5, stride=1, dilation=1, padding=2, bias=True)
        self.conv_1 = nn.Conv3d(self.g_dim,    self.g_dim*2,  5, stride=1, dilation=2, padding=4, bias=True)
        self.conv_2 = nn.Conv3d(self.g_dim*2,  self.g_dim*4,  5, stride=1, dilation=2, padding=4, bias=True)
        self.conv_3 = nn.Conv3d(self.g_dim*4,  self.g_dim*8,  5, stride=1, dilation=1, padding=2, bias=True)
        self.conv_4 = nn.Conv3d(self.g_dim*8,  self.g_dim*4,  5, stride=1, dilation=1, padding=2, bias=True)

        self.conv_5 = nn.ConvTranspose3d(self.g_dim*4,  self.g_dim*2, 4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim*2,  self.g_dim*2,  3, stride=1, padding=1, bias=True)
        self.conv_7 = nn.ConvTranspose3d(self.g_dim*2,  self.g_dim,   4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim,    1,             3, stride=1, padding=1, bias=True)

    def __init__(self, g_dim, sigmoid=False, alpha=1., init_weights=False):
        super(generator_adain, self).__init__(g_dim, sigmoid, alpha, init_weights)

    def forward(self, voxels_content, voxels_style, mask_, is_training=False):
        mask = F.interpolate(mask_, scale_factor=4, mode='nearest')

        out_content = self.conv_0(voxels_content)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)
        out_content = self.conv_1(out_content)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)
        out_content = self.conv_2(out_content)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)
        out_content = self.conv_3(out_content)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)
        out_content = self.conv_4(out_content)
        out_content = F.leaky_relu(out_content, negative_slope=0.02, inplace=True)

        out_style = self.conv_0(voxels_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        out_style = self.conv_1(out_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        out_style = self.conv_2(out_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        out_style = self.conv_3(out_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)
        out_style = self.conv_4(out_style)
        out_style = F.leaky_relu(out_style, negative_slope=0.02, inplace=True)

        t = adain_function.adaptive_instance_normalization(out_content, out_style)
        t = self.alpha * t + (1 - self.alpha) * out_content

        out = self.conv_5(t)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_8(out)

        #out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        #out = out.clamp(max=1.0)
        if self.sigmoid:
            out = torch.sigmoid(out)
        else:
            out = torch.max(torch.min(out, out*0.002+0.998), out*0.002)
        #out = torch.sigmoid(out)

        out = out*mask

        return out

