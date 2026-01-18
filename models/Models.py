import torch.nn as nn
from torch.nn import init
import torch
from models.dpcd_parts import (Conv_BN_ReLU, CGSU, Encoder_Block, Decoder_Block,
                               Changer_channel_exchange, log_feature,Heterogeneous_Dualbranch_Modulation_Fusion_Module)


class DPCD(nn.Module):
    def __init__(self):
        super().__init__()

        channel_list = [32, 64, 128, 256, 512]
        # encoder
        self.en_block1 = nn.Sequential(Conv_BN_ReLU(in_channel=3, out_channel=channel_list[0], kernel=3, stride=1),
                                       CGSU(in_channel=channel_list[0]),
                                       CGSU(in_channel=channel_list[0]),
                                       )
        self.en_block2 = Encoder_Block(in_channel=channel_list[0], out_channel=channel_list[1])
        self.en_block3 = Encoder_Block(in_channel=channel_list[1], out_channel=channel_list[2])
        self.en_block4 = Encoder_Block(in_channel=channel_list[2], out_channel=channel_list[3])
        self.en_block5 = Encoder_Block(in_channel=channel_list[3], out_channel=channel_list[4])

        self.channel_exchange4 = Changer_channel_exchange()

        # decoder
        self.de_block1 = Decoder_Block(in_channel=channel_list[4], out_channel=channel_list[3])
        self.de_block2 = Decoder_Block(in_channel=channel_list[3], out_channel=channel_list[2])
        self.de_block3 = Decoder_Block(in_channel=channel_list[2], out_channel=channel_list[1])

        self.dfc1 = Heterogeneous_Dualbranch_Modulation_Fusion_Module(in_d=channel_list[4], out_d=channel_list[4])
        self.dfc2 = Heterogeneous_Dualbranch_Modulation_Fusion_Module(in_d=channel_list[3], out_d=channel_list[3])
        self.dfc3 = Heterogeneous_Dualbranch_Modulation_Fusion_Module(in_d=channel_list[2], out_d=channel_list[2])
        self.dfc4 = Heterogeneous_Dualbranch_Modulation_Fusion_Module(in_d=channel_list[1], out_d=channel_list[1])
        # change path
        # the change block is the same as decoder block
        # the change block is used to fuse former and latter change features
        self.change_block4 = Decoder_Block(in_channel=channel_list[4], out_channel=channel_list[3])
        self.change_block3 = Decoder_Block(in_channel=channel_list[3], out_channel=channel_list[2])
        self.change_block2 = Decoder_Block(in_channel=channel_list[2], out_channel=channel_list[1])

        self.seg_out1 = nn.Conv2d(channel_list[1], 1, kernel_size=3, stride=1, padding=1)
        self.seg_out2 = nn.Conv2d(channel_list[1], 1, kernel_size=3, stride=1, padding=1)

        self.upsample_x2 = nn.Sequential(
            nn.Conv2d(channel_list[1], channel_list[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_list[0], 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.conv_out_change = nn.Conv2d(8, 1, kernel_size=7, stride=1, padding=3)


        # init parameters
        # using pytorch default init is enough
        # self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, t1, t2, log=False, img_name=None):
        
        # encoder
        t1_1 = self.en_block1(t1)
        t2_1 = self.en_block1(t2)

        if log:
            t1_2 = self.en_block2(t1_1, log=log, module_name='t1_1_en_block2', img_name=img_name)
            t2_2 = self.en_block2(t2_1, log=log, module_name='t2_1_en_block2', img_name=img_name)

            t1_3 = self.en_block3(t1_2, log=log, module_name='t1_2_en_block3', img_name=img_name)
            t2_3 = self.en_block3(t2_2, log=log, module_name='t2_2en_block3', img_name=img_name)

            t1_4 = self.en_block4(t1_3, log=log, module_name='t1_3_en_block4', img_name=img_name)
            t2_4 = self.en_block4(t2_3, log=log, module_name='t2_3_en_block4', img_name=img_name)
            t1_4, t2_4 = self.channel_exchange4(t1_4, t2_4)

            t1_5 = self.en_block5(t1_4, log=log, module_name='t1_4_en_block5', img_name=img_name)
            t2_5 = self.en_block5(t2_4, log=log, module_name='t2_4_en_block5', img_name=img_name)
        else:
            t1_2 = self.en_block2(t1_1)
            t2_2 = self.en_block2(t2_1)

            t1_3 = self.en_block3(t1_2)
            t2_3 = self.en_block3(t2_2)

            t1_4 = self.en_block4(t1_3)
            t2_4 = self.en_block4(t2_3)
            t1_4, t2_4 = self.channel_exchange4(t1_4, t2_4)

            t1_5 = self.en_block5(t1_4)
            t2_5 = self.en_block5(t2_4)

        de1_5 = t1_5
        de2_5 = t2_5

        de1_4 = self.de_block1(de1_5, t1_4)
        de2_4 = self.de_block1(de2_5, t2_4)

        de1_3 = self.de_block2(de1_4, t1_3)
        de2_3 = self.de_block2(de2_4, t2_3)

        de1_2 = self.de_block3(de1_3, t1_2)
        de2_2 = self.de_block3(de2_3, t2_2)

        seg_out1 = self.seg_out1(de1_2)
        seg_out2 = self.seg_out2(de2_2)

        if log:
            change_5 = self.dfc1(de1_5, de2_5, log=log, module_name='de1_5_de2_5_dfc1',
                                  img_name=img_name)

            change_4 = self.change_block4(change_5, self.dfc2(de1_4, de2_4, log=log, module_name='de1_4_de2_4_dfc2',
                                                               img_name=img_name))

            change_3 = self.change_block3(change_4, self.dfc3(de1_3, de2_3, log=log, module_name='de1_3_de2_3_dfc3',
                                                               img_name=img_name))

            change_2 = self.change_block2(change_3, self.dfc4(de1_2, de2_2, log=log, module_name='de1_2_de2_2_dfc4',
                                                               img_name=img_name))
        else:
            change_5 = self.dfc1(de1_5, de2_5)

            change_4 = self.change_block4(change_5, self.dfc2(de1_4, de2_4))

            change_3 = self.change_block3(change_4, self.dfc3(de1_3, de2_3))

            change_2 = self.change_block2(change_3, self.dfc4(de1_2, de2_2))

        change = self.upsample_x2(change_2)
        change_out = self.conv_out_change(change)

        if log:
            log_feature(log_list=[change_out, seg_out1, seg_out2], module_name='model',
                        feature_name_list=['change_out', 'seg_out1', 'seg_out2'],
                        img_name=img_name, module_output=False)

        return change_out, seg_out1, seg_out2

if __name__ == '__main__':
    net = DPCD().cuda()
    
    # 构造输入数据
    x1 = torch.randn(4, 3, 256, 256).cuda() 
    x2 = torch.randn(4, 3, 256, 256).cuda()   # 主干网络的输出特征 (对应 channels[0]=64)
    
    outputs = net(x1, x2)  # 正确传递 x 和 skips
    d3, d2, d1 = outputs  # 按返回值顺序解包
    
    # 打印输出形状
    print("d1.shape (最高层):", d1.shape)
    print("d2.shape:", d2.shape)
    print("d3.shape:", d3.shape)
