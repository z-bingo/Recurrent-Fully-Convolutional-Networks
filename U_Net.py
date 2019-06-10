import torch
import torch.nn as nn
import torch.nn.functional as F

"""
U-Net中连续两个Conv+BN+ReLU组合
"""
class Double_Conv(nn.Module):
    def __init__(self, in_channel, out_channel, sepConv=False):
        super(Double_Conv, self).__init__()
        if sepConv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, 3, 1, 1),
                nn.Conv2d(in_channel, out_channel, 1, 1, 0),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, 3, 1, 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, 3, 1, 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            )

    def forward(self, data):
        return self.conv(data)
# class Double_Conv(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(Double_Conv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, 3, 1, 1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channel, out_channel, 3, 1, 1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, data):
#         return self.conv(data)

class In_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(In_Conv, self).__init__()
        self.conv = Double_Conv(in_channel, out_channel, False)

    def forward(self, data):
        return self.conv(data)

class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = Double_Conv(in_channel, out_channel, False)

    def forward(self, *data):
        if len(data) == 1:
            return self.conv(self.pool(data[0]))
        # 第二项为已经下采样的部分
        elif len(data) == 2:
            data0 = self.pool(data[0])
            return self.conv(torch.cat([data0, data[1]], dim=1))

class Up(nn.Module):
    def __init__(self, in_channel_last, in_channel_toal, out_channel, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = F.interpolate
        else:
            self.up = nn.ConvTranspose2d(in_channel_last//2, in_channel_last//2, 2, stride=2)
        self.conv1 = nn.Conv2d(in_channel_last, in_channel_last//2, 3, 1, 1)
        self.conv = Double_Conv(in_channel_toal, out_channel)

    def forward(self, data_cur, data_pre, bilinear=True):
        if bilinear:
            data_cur = self.up(data_cur, scale_factor=2, mode='bilinear')
        else:
            data_cur = self.up(data_cur)
        data_cur = self.conv1(data_cur)
        # 从channel维度连接在一起
        return self.conv(torch.cat([data_cur, data_pre], dim=1))

class Out_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Out_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, data):
        return self.conv(data)
