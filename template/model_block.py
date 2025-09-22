import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from template.drop import drop_path


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x * self.relu6(x + 3.) / 6.

    def __repr__(self):
        return 'Hswish()'


class Act(nn.Module):
    def __init__(self, act_func='relu'):
        super(Act, self).__init__()
        if act_func == 'relu':
            self.act_func = nn.ReLU(inplace=True)
        elif act_func == 'h_swish':
            self.act_func = Hswish(inplace=True)
        elif act_func == 'none':
            self.act_func = nn.Identity()

    def forward(self, x):
        return self.act_func(x)


class SEBlock(nn.Module):
    def __init__(self, channels, se_ratio):
        super(SEBlock, self).__init__()

        squeeze_channels = channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')

        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.non_linear1 = nn.ReLU(inplace=True)
        self.se_expand = nn.Conv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_expand(y))
        y = x * y

        return y


class ConvBnAct(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, groups=1, act_func='relu'):
        super(ConvBnAct, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=kernel_size // 2, groups=groups, bias=False),
            nn.BatchNorm2d(C_out),
            Act(act_func=act_func),
        )

    def forward(self, x):
        out = self.op(x)
        return out


class MobileNetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, expansion_factor=6, stride=1, se_ratio=0.25, act_func='h_swish', no_reslink=False):
        super(MobileNetBottleneck, self).__init__()
        self.no_reslink = no_reslink
        expansion_channels = int(in_channels * expansion_factor)

        self.op1 = ConvBnAct(in_channels, expansion_channels, kernel_size=1, stride=1, act_func=act_func)
        self.op2 = ConvBnAct(expansion_channels, expansion_channels, kernel_size=kernel_size, stride=stride, groups=expansion_channels, act_func='none')
        # self.se = SEBlock(expansion_channels, se_ratio=se_ratio) if se_ratio != 0.0 else nn.Identity()
        self.op2_act = Act(act_func=act_func)
        self.op3 = ConvBnAct(expansion_channels, out_channels, kernel_size=1, stride=1, act_func='none')
        self.se = SEBlock(out_channels, se_ratio=se_ratio) if se_ratio != 0.0 else nn.Identity()

        # self.shortcut = (stride == 1 and in_channels == out_channels)
        if (in_channels == out_channels and stride == 1) or no_reslink:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                ConvBnAct(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels, act_func='none'),
                ConvBnAct(in_channels, out_channels, kernel_size=1, stride=1, act_func='none'),
            )

    def forward(self, x, drop_path_rate=0.0):
        # B x c x h x w
        out = self.op1(x)
        out = self.op2(out)
        # out = self.se(out)
        out = self.op2_act(out)
        out = self.op3(out)
        out = self.se(out)

        if self.shortcut and not self.no_reslink:
            if drop_path_rate > 0:
                out = drop_path(out, drop_prob=drop_path_rate, training=self.training)
            return out + x
        else:
            return out
        # if not self.no_reslink:
        #     if drop_path_rate > 0:
        #         out = drop_path(out, drop_prob=drop_path_rate, training=self.training)
        #     return out + self.shortcut(x)
        # else:
        #     return out


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dw_size=3, ratio=2, stride=1, act_func='h_swish'):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = ConvBnAct(in_channels, init_channels, kernel_size=kernel_size, stride=stride, act_func=act_func)
        self.cheap_conv = ConvBnAct(init_channels, new_channels, kernel_size=dw_size, stride=1, groups=init_channels, act_func=act_func)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_conv(x1)

        out = torch.cat([x1, x2], dim=1)
        # 输出需要的维度数量
        return out[:, :self.out_channels, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, expansion_factor=6, stride=1, compress_ratio=2, se_ratio=0.25, act_func='h_swish', no_reslink=False):
        super(GhostBottleneck, self).__init__()

        self.no_reslink = no_reslink
        expansion_channels = int(in_channels * expansion_factor)

        self.ghost1 = GhostModule(in_channels, expansion_channels, dw_size=kernel_size, ratio=compress_ratio, act_func=act_func)
        self.dw_conv = ConvBnAct(expansion_channels, expansion_channels, kernel_size=3, stride=stride, groups=expansion_channels, act_func='none') if stride == 2 else nn.Identity()
        # self.se = SEBlock(expansion_channels, se_ratio=se_ratio) if se_ratio != 0.0 else nn.Identity()
        self.ghost2 = GhostModule(expansion_channels, out_channels, dw_size=kernel_size, ratio=compress_ratio, act_func='none')
        self.se = SEBlock(out_channels, se_ratio=se_ratio) if se_ratio != 0.0 else nn.Identity()

        # 调整通道数和特征图大小
        if (in_channels == out_channels and stride == 1) or no_reslink:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                ConvBnAct(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels, act_func='none'),
                ConvBnAct(in_channels, out_channels, kernel_size=1, stride=1, act_func='none'),
            )

    def forward(self, x, drop_path_rate=0.0):
        out = self.ghost1(x)
        out = self.dw_conv(out)
        # out = self.se(out)
        out = self.ghost2(out)
        out = self.se(out)

        if not self.no_reslink:
            if drop_path_rate > 0:
                out = drop_path(out, drop_prob=drop_path_rate, training=self.training)
            return out + self.shortcut(x)
        else:
            return out
