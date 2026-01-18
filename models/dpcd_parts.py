import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from utils.path_hyperparameter import ph
import cv2
from torchvision import transforms as T
from pathlib import Path


class Conv_BN_ReLU(nn.Module):
    """ Basic convolution."""

    def __init__(self, in_channel, out_channel, kernel, stride):
        super().__init__()
        self.conv_bn_relu = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
                                                    padding=kernel // 2, bias=False, stride=stride),
                                          nn.BatchNorm2d(out_channel),
                                          nn.ReLU(inplace=True),
                                          )

    def forward(self, x):
        output = self.conv_bn_relu(x)

        return output


class CGSU(nn.Module):
    """Basic convolution module."""

    def __init__(self, in_channel):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1,
                                             bias=False),
                                   nn.BatchNorm2d(in_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=ph.dropout_p)
                                   )

    def forward(self, x):
        x = self.conv1(x)
        return x


class CGSU_DOWN(nn.Module):
    """Basic convolution module with stride=2."""

    def __init__(self, in_channel):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1,
                                             stride=2, bias=False),
                                   nn.BatchNorm2d(in_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=ph.dropout_p)
                                   )
        self.conv_res = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # remember the tensor should be contiguous
        output1 = self.conv1(x)

        # respath
        output2 = self.conv_res(x)

        output = torch.cat([output1, output2], dim=1)

        return output


class Changer_channel_exchange(nn.Module):
    """Exchange channels of two feature uniformly-spaced with 1:1 ratio."""

    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x1, x2):
        N, C, H, W = x1.shape
        exchange_mask = torch.arange(C) % self.p == 0
        exchange_mask1 = exchange_mask.cuda().int().expand((N, C)).unsqueeze(-1).unsqueeze(-1)  # b,c,1,1
        exchange_mask2 = 1 - exchange_mask1
        out_x1 = exchange_mask1 * x1 + exchange_mask2 * x2
        out_x2 = exchange_mask1 * x2 + exchange_mask2 * x1

        return out_x1, out_x2

class CBAM(nn.Module):
    """Attention module."""

    def __init__(self, in_channel):
        super().__init__()
        self.k = kernel_size(in_channel)
        self.channel_conv = nn.Conv1d(2, 1, kernel_size=self.k, padding=self.k // 2)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x, log=False, module_name=None, img_name=None):
        avg_channel = self.avg_pooling(x).squeeze(-1).transpose(1, 2)  # b,1,c
        max_channel = self.max_pooling(x).squeeze(-1).transpose(1, 2)  # b,1,c
        channel_weight = self.channel_conv(torch.cat([avg_channel, max_channel], dim=1))
        channel_weight = self.sigmoid(channel_weight).transpose(1, 2).unsqueeze(-1)  # b,c,1,1
        x = channel_weight * x

        avg_spatial = torch.mean(x, dim=1, keepdim=True)  # b,1,h,w
        max_spatial = torch.max(x, dim=1, keepdim=True)[0]  # b,1,h,w
        spatial_weight = self.spatial_conv(torch.cat([avg_spatial, max_spatial], dim=1))  # b,1,h,w
        spatial_weight = self.sigmoid(spatial_weight)
        output = spatial_weight * x

        if log:
            log_list = [spatial_weight]
            feature_name_list = ['spatial_weight']
            log_feature(log_list=log_list, module_name=module_name,
                        feature_name_list=feature_name_list,
                        img_name=img_name, module_output=True)

        return output

#---------------------------------------------------------------------------------------------------------------
def cosine_similarity(x1, x2, eps=1e-8):
    dot_product = torch.sum(x1 * x2, dim=1)
    norm_x1 = torch.norm(x1, p=2, dim=1)
    norm_x2 = torch.norm(x2, p=2, dim=1)
    max_norm = torch.max(norm_x1 * norm_x2, torch.tensor(eps).to(x1.device))
    cos_sim = dot_product / max_norm
    return cos_sim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class AFFM(nn.Module):
    def __init__(self, channels, attn_kernel=7, cutoff_ratio=0.1, device='cuda'):
        super().__init__()
        # 通道扩2倍（拼接实部/虚部）
        self.conv1x1 = nn.Conv2d(2*channels, 2*channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(2*channels)
        self.prelu = nn.PReLU()
        self.spatial_attn = nn.Conv2d(2, 1, kernel_size=attn_kernel, padding=attn_kernel//2)
        self.sigmoid = nn.Sigmoid()
        self.cutoff_ratio = cutoff_ratio
        self.device = device
        
    def get_frequency_masks(self, H, W, device):
        y, x = np.ogrid[:H, :W]
        center_y, center_x = H // 2, W // 2
        radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_radius = np.max(radius)
        low_mask = (radius <= max_radius * self.cutoff_ratio).astype(np.float32)
        high_mask = 1.0 - low_mask
        # [1,1,H,W]方便广播
        return (torch.from_numpy(low_mask)[None, None, ...].to(device),
                torch.from_numpy(high_mask)[None, None, ...].to(device))

    @staticmethod
    def negative_cosine_attention(f1, f2, eps=1e-8):
        # f1, f2: [B, C, H, W] (实部/虚部拼通道)
        f1_norm = F.normalize(f1, p=2, dim=1, eps=eps)
        f2_norm = F.normalize(f2, p=2, dim=1, eps=eps)
        cos_sim = torch.sum(f1_norm * f2_norm, dim=1, keepdim=True)
        neg_cos_att = (1 - cos_sim) / 2
        return neg_cos_att

    @staticmethod
    def cosine_attention(f1, f2, eps=1e-8):
        f1_norm = F.normalize(f1, p=2, dim=1, eps=eps)
        f2_norm = F.normalize(f2, p=2, dim=1, eps=eps)
        cos_sim = torch.sum(f1_norm * f2_norm, dim=1, keepdim=True)
        att = (1 + cos_sim) / 2
        return att

    def forward(self, f1, f2):
        """
        f1, f2: [B, C, H, W]  # feature maps
        Returns:
            out1, out2: [B, C, H, W] # enhanced feature maps
        """
        B, C, H, W = f1.shape
        f1 = f1.float()
        f2 = f2.float()
        # 1. 2D FFT
        fft1 = torch.fft.fft2(f1, norm='ortho')
        fft2 = torch.fft.fft2(f2, norm='ortho')
        
        # Step 2. 获得低/高频mask
        low_mask, high_mask = self.get_frequency_masks(H, W, device=f1.device)
        # Step 3. 高/低频分支提取
        # [B,C,H,W]复数 × [1,1,H,W] 实现广播
        T1_fft_low = fft1 * low_mask
        T2_fft_low = fft2 * low_mask
        T1_fft_high = fft1 * high_mask
        T2_fft_high = fft2 * high_mask

        # 2. 拼接实部和虚部
        def split_complex(t):  # t: [B, C, H, W] 复数
            return torch.cat([t.real, t.imag], dim=1)

        T1_low_cat = split_complex(T1_fft_low)
        T2_low_cat = split_complex(T2_fft_low)
        T1_high_cat = split_complex(T1_fft_high)
        T2_high_cat = split_complex(T2_fft_high)

        # Step 5. 注意力权重计算
        att_low = self.negative_cosine_attention(T1_low_cat, T2_low_cat)     # 风格差异大权重低
        # Step 6. 注意力加权
        T1_fft_low_weighted = T1_low_cat * att_low+T1_low_cat
        T2_fft_low_weighted = T2_low_cat * att_low+T2_low_cat
        
        # 3. 差分 + 1x1卷积 + BN + PReLU
        
        diff = torch.abs(T1_high_cat - T2_high_cat)
        diff = self.conv1x1(diff)
        diff = self.bn(diff)
        diff = self.prelu(diff)  # [B, 2C, H, W]

        # 4. max/avg池化拼接
        max_pool, _ = torch.max(diff, dim=1, keepdim=True)  # [B, 1, H, W]
        avg_pool = torch.mean(diff, dim=1, keepdim=True)    # [B, 1, H, W]
        pool = torch.cat([max_pool, avg_pool], dim=1)       # [B, 2, H, W]
        attn_map = self.spatial_attn(pool)                  # [B, 1, H, W]
        attn_map = self.sigmoid(attn_map)                   # [B, 1, H, W]

        # 5. 注意力加权（channel维自动广播）
        T1_fft_high_weighted = T1_high_cat * attn_map+T1_high_cat
        T2_fft_high_weighted = T2_high_cat * attn_map+T2_high_cat
        
        T1_fft_fused = T1_fft_low_weighted + T1_fft_high_weighted
        T2_fft_fused = T2_fft_low_weighted + T2_fft_high_weighted
        

        # 6. 拆分回复数并iFFT
        def complex_recover(cat):
            real, imag = torch.chunk(cat, 2, dim=1)  # [B,C,H,W]各自
            return torch.complex(real, imag)

        fft1_new = complex_recover(T1_fft_fused)
        fft2_new = complex_recover(T2_fft_fused)

        # 7. 逆FFT回到时域
        out1 = torch.fft.ifft2(fft1_new, norm='ortho').real  # 取实部
        out2 = torch.fft.ifft2(fft2_new, norm='ortho').real

        return out1, out2


class Heterogeneous_Dualbranch_Modulation_Fusion_Module(nn.Module):
    def __init__(self, in_d, out_d):
        super(Heterogeneous_Dualbranch_Modulation_Fusion_Module, self).__init__()
        #self.channel_att = nn.Sequential(SELayer(768, 256), SELayer(768, 256), SELayer(768, 256), SELayer(768, 256))
        self.in_d = in_d
        self.out_d = out_d
        self.conv_sub = nn.Sequential(
            nn.Conv2d(in_d, in_d, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_d),
            nn.ReLU(inplace=True)
        )
        # self.fpn = FeaturePyramidNetwork(in_d * 2)  # 处理两个输入特征图的合并
        #self.channel_attention = ChannelAttentionLayer(in_d * 3)  # 根据需要调整 num_heads
        self.channel_attention = CBAM(in_d * 2)
        # Assuming that the concatenated channel count is 1536
        # self.dpfa = DPFA(in_d)
        self.conv_dr = nn.Sequential(
            nn.Conv2d(in_d*2, out_d, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_d),
            nn.ReLU(inplace=True)
        )
        self.conv_dr_1 = nn.Sequential(
            nn.Conv2d(in_d, out_d, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_d),
            nn.ReLU(inplace=True)
        )
        self.conv_diff_enh1 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_diff_enh2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        
        self.block = nn.Sequential(
            nn.Conv2d(in_d*2, in_d, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_d),
            nn.ReLU(inplace=True)
        )
        self.sig = nn.Sigmoid()  
        self.affm = AFFM(channels=in_d)

    def forward(self, x1, x2, log=None, module_name=None,img_name=None):
        x1, x2 = self.affm(x1, x2)
        x=torch.cat([x1, x2], dim=1)
        c1 = self.sig(F.adaptive_max_pool2d(x, (1, 1)))
        c2 = self.sig(F.adaptive_avg_pool2d(x, (1, 1)))
        # 特征融合
        x=c1 * x + c2 * x
        Fsub = torch.abs(x1 - x2)
        Fsub = self.conv_sub(Fsub)
        cos_sim = cosine_similarity(x1, x2)
        m = (1 - cos_sim).unsqueeze(1)
        concatenated_features = torch.cat([x1, x2], dim=1)
        Fpn_feature = self.block(concatenated_features)
        # 注意力权重
        σ1 = self.sig(F.adaptive_max_pool2d(Fpn_feature, (1, 1)))
        σ2 = self.sig(F.adaptive_avg_pool2d(Fpn_feature, (1, 1)))
        # 特征融合
        Fpn_feature=σ1 * Fpn_feature + σ2 * Fpn_feature
        Fpn_m=Fpn_feature * m
        #print("---------------------------------------------------------------------------------------------Fpn_m:",Fpn_m.shape) 
        Fpn_cat=torch.cat([Fpn_m, Fsub], dim=1)
        enhanced_feature = self.channel_attention(Fpn_cat)
        output=enhanced_feature*x
        output = self.conv_dr(output)
        return output
#-------------------------------------------------------------------------------------------------------------------------------    
    
class Encoder_Block(nn.Module):
    """ Basic block in encoder"""

    def __init__(self, in_channel, out_channel):
        super().__init__()

        assert out_channel == in_channel * 2, 'the out_channel is not in_channel*2 in encoder block'
        self.conv1 = nn.Sequential(
            CGSU_DOWN(in_channel=in_channel),
            CGSU(in_channel=out_channel),
            CGSU(in_channel=out_channel)
        )
        self.conv3 = Conv_BN_ReLU(in_channel=out_channel, out_channel=out_channel, kernel=3, stride=1)
        self.cbam = CBAM(in_channel=out_channel)

    def forward(self, x, log=False, module_name=None, img_name=None):
        x = self.conv1(x)
        x = self.conv3(x)
        x_res = x.clone()
        if log:
            output = self.cbam(x, log=log, module_name=module_name + '-x_cbam', img_name=img_name)
        else:
            output = self.cbam(x)
        output = x_res + output

        return output


class Decoder_Block(nn.Module):
    """Basic block in decoder."""

    def __init__(self, in_channel, out_channel):
        super().__init__()

        assert out_channel == in_channel // 2, 'the out_channel is not in_channel//2 in decoder block'
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse = nn.Sequential(nn.Conv2d(in_channels=in_channel + out_channel, out_channels=out_channel,
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(inplace=True),
                                  )

    def forward(self, de, en):
        de = self.up(de)
        output = torch.cat([de, en], dim=1)
        output = self.fuse(output)

        return output


# from ECANet, in which y and b is set default to 2 and 1
def kernel_size(in_channel):
    """Compute kernel size for one dimension convolution in eca-net"""
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k


def channel_split(x):
    """Half segment one feature on channel dimension into two features, mixture them on channel dimension,
    and split them into two features."""
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


def log_feature(log_list, module_name, feature_name_list, img_name, module_output=True):
    """ Log output feature of module and model

    Log some output features in a module. Feature in :obj:`log_list` should have corresponding name
    in :obj:`feature_name_list`.

    For module output feature, interpolate it to :math:`ph.patch_size`×:math:`ph.patch_size`,
    log it in :obj:`cv2.COLORMAP_JET` format without other change,
    and log it in :obj:`cv2.COLORMAP_JET` format with equalization.
    For model output feature, log it without any change.

    Notice that feature is log in :obj:`ph.log_path`/:obj:`module_name`/
    name in :obj:`feature_name_list`/:obj:`img_name`.jpg.

    Parameter:
        log_list(list): list of output need to log.
        module_name(str): name of module which output the feature we log,
            if :obj:`module_output` is False, :obj:`module_name` equals to `model`.
        module_output(bool): determine whether output is from module or model.
        feature_name_list(list): name of output feature.
        img_name(str): name of corresponding image to output.


    """
    for k, log in enumerate(log_list):
        log = log.clone().detach()
        b, c, h, w = log.size()
        if module_output:
            log = torch.mean(log, dim=1, keepdim=True)
            log = F.interpolate(
                log * 255, scale_factor=ph.patch_size // h,
                mode='nearest').reshape(b, ph.patch_size, ph.patch_size, 1) \
                .cpu().numpy().astype(np.uint8)
            log_dir = ph.log_path + module_name + '/' + feature_name_list[k] + '/'
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_equalize_dir = ph.log_path + module_name + '/' + feature_name_list[k] + '_equalize/'
            Path(log_equalize_dir).mkdir(parents=True, exist_ok=True)

            for i in range(b):
                log_i = cv2.applyColorMap(log[i], cv2.COLORMAP_JET)
                cv2.imwrite(log_dir + img_name[i] + '.jpg', log_i)

                log_i_equalize = cv2.equalizeHist(log[i])
                log_i_equalize = cv2.applyColorMap(log_i_equalize, cv2.COLORMAP_JET)
                cv2.imwrite(log_equalize_dir + img_name[i] + '.jpg', log_i_equalize)
        else:
            log_dir = ph.log_path + module_name + '/' + feature_name_list[k] + '/'
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log = torch.round(torch.sigmoid(log))
            log = F.interpolate(log, scale_factor=ph.patch_size // h,
                                mode='nearest').cpu()
            to_pil_img = T.ToPILImage(mode=None)
            for i in range(b):
                log_i = to_pil_img(log[i])
                log_i.save(log_dir + img_name[i] + '.jpg')
