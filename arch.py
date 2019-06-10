import torch
import torch.nn as nn
from U_Net import *

"""
单帧去噪采用U-Net结构
"""
class Single_Frame_Net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Single_Frame_Net, self).__init__()
        self.inc = In_Conv(in_channel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 1024, 512)
        self.up2 = Up(512, 512, 256)
        self.up3 = Up(256, 256, 128)
        self.up4 = Up(128, 128, 64)
        self.outc = Out_Conv(64, out_channel)

    def forward(self, data):
        """
        The forward function of the SFN network.
        :param data: one frame of the input burst
        :return: the output of SFN and its features of each layers
        """
        f1 = self.inc(data)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)
        f6 = self.up1(f5, f4)
        f7 = self.up2(f6, f3)
        f8 = self.up3(f7, f2)
        f9 = self.up4(f8, f1)
        # feature是要传入Multi_Frame_Net的每层的特征
        feature = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
        out = self.outc(f9)
        return out, feature

"""
多帧网络是基于时间维度
每层的输入为：
    当前时刻： SFN对应层的输出
    前一时刻： MFN对应层的输出
"""
class Multi_Frame_Net(nn.Module):
    def __init__(self, out_channel):
        super(Multi_Frame_Net, self).__init__()
        self.inc = In_Conv(64 + 64, 64)  # SFN.inc + MFN.inc(t-1)
        self.down1 = Down(64 + 128 + 128, 128)   #
        self.down2 = Down(128 + 256 + 256, 256)
        self.down3 = Down(256 + 512 + 512, 512)
        self.down4 = Down(512 + 1024 + 1024, 1024)
        self.up1 = Up(1024, 1024 + 512 + 512, 512)
        self.up2 = Up(512, 512 + 256 + 256, 256)
        self.up3 = Up(256, 256 + 128 + 128, 128)
        self.up4 = Up(128, 128 + 64 + 64, 64)
        self.outc = Out_Conv(64, out_channel)

    def forward(self, *input):
        """
        The forward function of MFN
        :param input: one or two, if one, it is the feature of SFN at time instance 1;
                    otherwise, the inputs are features of SFN and MFN, where MFN is the last time instance.
        :return: the output of this MFN and its features of each layers
        """
        if len(input) == 1:
            f_sfn = input[0]
            device = f_sfn[0].device
            f1 = self.inc(torch.cat([f_sfn[0], torch.zeros(f_sfn[0].size(), device=device)], dim=1))
            f2 = self.down1(f1, torch.cat([f_sfn[1], torch.zeros(f_sfn[1].size(), device=device)], dim=1))
            f3 = self.down2(f2, torch.cat([f_sfn[2], torch.zeros(f_sfn[2].size(), device=device)], dim=1))
            f4 = self.down3(f3, torch.cat([f_sfn[3], torch.zeros(f_sfn[3].size(), device=device)], dim=1))
            f5 = self.down4(f4, torch.cat([f_sfn[4], torch.zeros(f_sfn[4].size(), device=device)], dim=1))
            f6 = self.up1(f5, torch.cat([f_sfn[5], torch.zeros(f_sfn[5].size(), device=device), f4], dim=1))
            f7 = self.up2(f6, torch.cat([f_sfn[6], torch.zeros(f_sfn[6].size(), device=device), f3], dim=1))
            f8 = self.up3(f7, torch.cat([f_sfn[7], torch.zeros(f_sfn[7].size(), device=device), f2], dim=1))
            f9 = self.up4(f8, torch.cat([f_sfn[8], torch.zeros(f_sfn[8].size(), device=device), f1], dim=1))
            feature = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
        else:
            f_sfn, f_mfn = input
            f1 = self.inc(torch.cat([f_sfn[0], f_mfn[0]], dim=1))
            f2 = self.down1(f1, torch.cat([f_sfn[1], f_mfn[1]], dim=1))
            f3 = self.down2(f2, torch.cat([f_sfn[2], f_mfn[2]], dim=1))
            f4 = self.down3(f3, torch.cat([f_sfn[3], f_mfn[3]], dim=1))
            f5 = self.down4(f4, torch.cat([f_sfn[4], f_mfn[4]], dim=1))
            f6 = self.up1(f5, torch.cat([f_sfn[5], f_mfn[5], f4], dim=1))
            f7 = self.up2(f6, torch.cat([f_sfn[6], f_mfn[6], f3], dim=1))
            f8 = self.up3(f7, torch.cat([f_sfn[7], f_mfn[7], f2], dim=1))
            f9 = self.up4(f8, torch.cat([f_sfn[8], f_mfn[8], f1], dim=1))
            feature = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
            # Output
        return self.outc(f9), feature

"""
The full architecture of RNN based FCN
"""
class Deep_Burst_Denoise(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Deep_Burst_Denoise, self).__init__()
        self.SFN = Single_Frame_Net(in_channel, out_channel)
        self.MFN = Multi_Frame_Net(out_channel)

    def forward(self, *input):
        """
        如果是第一个时刻，只输入当前时刻的数据
        否则，还要同时输入mfn上一时刻的数据
        :param input:
        :return:
        """
        if len(input) == 1:
            data = input[0]
            sfn_out, sfn_f = self.SFN(data)
            mfn_out, mfn_f = self.MFN(sfn_f)
        else:
            data, mfn_f_last = input
            sfn_out, sfn_f = self.SFN(data)
            mfn_out, mfn_f = self.MFN(sfn_f, mfn_f_last)
        return sfn_out, mfn_out, mfn_f


if __name__ == '__main__':
    from tensorboardX import SummaryWriter
    from torchsummary import summary
    dbn = Deep_Burst_Denoise(3, 3).cuda()
    # inp = torch.rand((1, 3, 224, 224)).cuda()
    # with SummaryWriter(comment='SFN') as arch_writer:
    #     # _, f = sfn(inp)
    #     arch_writer.add_graph(dbn, inp)

    summary(dbn, (3,224,224), batch_size=1, device='cuda')