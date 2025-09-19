import torch
import torch.nn.functional as F
import torchvision
from model.submodules import *


def resize_2d(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return F.interpolate(inputs, [h, w], mode=mode, align_corners=True)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        squeeze = F.adaptive_avg_pool2d(x, (1, 1))  # (batch_size, channels, 1, 1)
        squeeze = squeeze.view(squeeze.size(0), -1)  # (batch_size, channels)

        excitation = self.fc1(squeeze)
        excitation = F.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)

        excitation = excitation.view(excitation.size(0), excitation.size(1), 1, 1)
        return x * excitation


class FusionNet(nn.Module):
    def __init__(self, cond_c):
        super(FusionNet, self).__init__()
        self.conv1 = conv3x3_leaky_relu(cond_c[-1] * 2, cond_c[-1])
        self.conv2 = conv3x3_leaky_relu(cond_c[-2] * 2, cond_c[-2])
        self.conv3 = conv3x3_leaky_relu(cond_c[-3] * 2, cond_c[-3])

        self.se1 = SEBlock(cond_c[-1])
        self.se2 = SEBlock(cond_c[-2])
        self.se3 = SEBlock(cond_c[-3])

    def forward(self, F0, F1):
        x1 = self.conv1(torch.cat((F0[-1], F1[-1]), 1))
        x1 = x1 + self.se1(x1)

        x2 = self.conv2(torch.cat((F0[-2], F1[-2]), 1))
        x2 = x2 + self.se2(x2)

        x3 = self.conv3(torch.cat((F0[-3], F1[-3]), 1))
        x3 = x3 + self.se3(x3)
        return x3, x2, x1


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1)
    )



class OffsetEstimator(nn.Module):
    def __init__(self, ch_in_frame, ch_in_event, prev_offset=False):
        super(OffsetEstimator, self).__init__()
        nf = ch_in_frame * 2 + ch_in_event
        self.conv0 = nn.Sequential(
            conv1x1(nf, nf),
            nn.ReLU(),
            conv_resblock_one(nf, nf),
            conv_resblock_one(nf, nf),
        )
        num = 2*3*3
        if prev_offset:
            na = nf + num
        else:
            na = nf
        self.conv2 = nn.Sequential(
            conv3x3(na, nf),
            nn.ReLU()
        )
        self.offset = conv3x3(nf, num)
        self.mask = conv3x3(nf, num // 2)

    def forward(self, x, offset=None):
        x = self.conv0(x)
        if offset is not None:
            x = torch.cat((x, offset), 1)
        x = self.conv2(x)
        offset = self.offset(x)
        mask = self.mask(x)
        return offset, mask



class FeaTNet(nn.Module):
    def __init__(self, bins):
        super(FeaTNet, self).__init__()
        num_chs_frame = [3, 16, 32, 64, 128]
        num_chs_event = [bins, 16, 32, 64, 96]
        num_chs_ref = [1, 8, 16, 32, 64]
        self.frame_encoder = EncoderImage(num_chs_frame)
        self.event_encoder = EncoderEvent(num_chs_event)
        self.ref_encoder = EncoderRef(num_chs_ref)
        self.predict_flow = nn.ModuleList([conv3x3_leaky_relu(num_chs_event[-1], 2),
                                           conv3x3_leaky_relu(num_chs_event[-2], 2),
                                           conv3x3_leaky_relu(num_chs_event[-3], 2)])
        self.offset_estimator = nn.ModuleList([
            OffsetEstimator(num_chs_frame[-1], num_chs_event[-1], num_chs_ref[-1], prev_offset=False),
            OffsetEstimator(num_chs_frame[-2], num_chs_event[-2], num_chs_ref[-2], prev_offset=True),
            OffsetEstimator(num_chs_frame[-3], num_chs_event[-3], num_chs_ref[-3], prev_offset=True)
        ])
        self.deform_conv = nn.ModuleList([
            torchvision.ops.DeformConv2d(num_chs_frame[-1], num_chs_frame[-2], 3, 1, 1),
            torchvision.ops.DeformConv2d(num_chs_frame[-2], num_chs_frame[-3], 3, 1, 1),
            torchvision.ops.DeformConv2d(num_chs_frame[-3], num_chs_frame[-3], 3, 1, 1)
        ])
        self.ref_proj_layers = nn.ModuleList([
            conv3x3(num_chs_ref[-2], num_chs_ref[-2]),
            conv3x3(num_chs_ref[-3], num_chs_ref[-3]),
        ])
        self.fusion_block = nn.ModuleList([
            conv_resblock_one(num_chs_frame[-2] + num_chs_frame[-3] + num_chs_ref[-2], num_chs_frame[-2]),
            conv_resblock_one(num_chs_frame[-2] + num_chs_frame[-3] + num_chs_ref[-3], num_chs_frame[-3])
        ])
        self.lastconv = nn.ConvTranspose2d(num_chs_frame[-3], num_chs_frame[-3], 4, 2, 1)

    def forward(self, img0, img1, v0, rec):
        F0_2, F0_1, F0_0 = self.frame_encoder(img0)
        F1_2, F1_1, F1_0 = self.frame_encoder(img1)
        E0_2, E0_1, E0_0 = self.event_encoder(v0)
        R_2, R_1, R_0 = self.ref_encoder(rec)
        _, _, H, W = img0.shape

        # ------0------
        feat_t_in = torch.cat((F0_0, E0_0, R_0, F1_0), 1)     # B, 513
        off_0, m_0 = self.offset_estimator[0](feat_t_in)
        F0_0_ = self.deform_conv[0](F0_0, off_0, m_0)
        off_0_up = resize_2d(off_0, F0_1)

        # ------1-------
        feat_t_in = torch.cat((F0_1, E0_1, R_1, F1_1), 1)
        off_1, m_1 = self.offset_estimator[1](feat_t_in, off_0_up)
        F0_1_ = self.deform_conv[1](F0_1, off_1, m_1)
        off_1_up = resize_2d(off_1, F0_2)
        F0_0_up = resize_2d(F0_0_, F0_1)
        ref_1 = self.ref_proj_layers[0](R_1)
        F0_1_ = self.fusion_block[0](torch.cat((F0_1_, F0_0_up, ref_1), 1))

        # ------2-------
        feat_t_in = torch.cat((F0_2, E0_2, R_2, F1_2), 1)
        off_2, m_2 = self.offset_estimator[2](feat_t_in, off_1_up)
        F0_2_ = self.deform_conv[2](F0_2, off_2, m_2)
        F0_1_up = resize_2d(F0_1_, F0_2)
        ref_2 = self.ref_proj_layers[1](R_2)
        F0_2_ = self.fusion_block[1](torch.cat((F0_2_, F0_1_up, ref_2), 1))

        out = self.lastconv(F0_2_)
        return out


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim // 2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class EncoderImage(nn.Module):
    def __init__(self, num_chs):
        super(EncoderImage, self).__init__()
        self.conv0_img = conv3x3_resblock_one(3, num_chs[0] // 2, stride=1)
        
        # feat 1/2
        self.conv1_img = conv3x3_resblock_one(num_chs[0] // 2, num_chs[0], stride=2)
        self.conv1_dino = conv(64, num_chs[0], stride=1)
        self.ffn1 = FeedForward(num_chs[0] * 2, ffn_expansion_factor=2.66, bias=False)

        # feat 1/4
        self.conv2_img = conv3x3_resblock_one(num_chs[0], num_chs[1], stride=2)
        self.conv2_dino = conv(256, num_chs[1], stride=1)
        self.ffn2 = FeedForward(num_chs[1] * 2, ffn_expansion_factor=2.66, bias=False)

        # feat 1/8
        self.conv3_img = conv3x3_resblock_one(num_chs[1], num_chs[2], stride=2)
        self.conv3_dino = conv(512, num_chs[2], stride=1)
        self.ffn3 = FeedForward(num_chs[2] * 2, ffn_expansion_factor=2.66, bias=False)
        
        self.tanh = nn.Tanh()

    def forward(self, img, dino_feat):
        # dino_feat: B,64,128,128\ B,256,64,64\ B,512,32,32\ B,1024,16,16
        feat_img_ = self.conv0_img(img)
        
        # 1/2
        feat_0_img = self.conv1_img(feat_img_)
        feat_0_dino = self.conv1_dino(dino_feat[0])
        feat_0 = self.ffn1(torch.cat((feat_0_img, feat_0_dino), 1))
        feat_0 = self.tanh(feat_0)

        # 1/4
        feat_1_img = self.conv2_img(feat_0_img)
        feat_1_dino = self.conv2_dino(dino_feat[1])
        feat_1 = self.ffn2(torch.cat((feat_1_img, feat_1_dino), 1))
        feat_1 = self.tanh(feat_1)

        # 1/8
        feat_2_img = self.conv3_img(feat_1_img)
        feat_2_dino = self.conv3_dino(dino_feat[2])
        feat_2 = self.ffn3(torch.cat((feat_2_img, feat_2_dino), 1))
        feat_2 = self.tanh(feat_2)
        return feat_0, feat_1, feat_2


class EncoderEvent(nn.Module):
    def __init__(self, bins, num_chs):
        super(EncoderEvent, self).__init__()
        self.conv1 = conv3x3_resblock_one(bins, num_chs[0], stride=1)
        self.conv2 = conv3x3_resblock_one(num_chs[0], num_chs[0], stride=2)
        self.conv3 = conv3x3_resblock_one(num_chs[0], num_chs[1], stride=2)
        self.conv4 = conv3x3_resblock_one(num_chs[1], num_chs[2], stride=2)

    def forward(self, events):
        x = self.conv1(events)
        x12 = self.conv2(x)
        x14 = self.conv3(x12)
        x18 = self.conv4(x14)
        return x12, x14, x18
    
class alignNet(nn.Module):
    def __init__(self, bins, num_chs):
        super(alignNet, self).__init__()
        self.event_encoder = EncoderEvent(bins, num_chs)
        self.fit_conv = nn.ModuleList([
            conv3x3_leaky_relu(num_chs[-1] * 2, num_chs[-1]),
            conv3x3_leaky_relu(num_chs[-2] * 2, num_chs[-2]),
            conv3x3_leaky_relu(num_chs[-3] * 2, num_chs[-3]),
        ])
        self.offset_estimator = nn.ModuleList([
            OffsetEstimator(num_chs[-1], num_chs[-1], prev_offset=False),
            OffsetEstimator(num_chs[-2], num_chs[-2], prev_offset=True),
            OffsetEstimator(num_chs[-3], num_chs[-3], prev_offset=True)
        ])
        self.deform_conv = nn.ModuleList([
            torchvision.ops.DeformConv2d(num_chs[-1], num_chs[-2], 3, 1, 1),
            torchvision.ops.DeformConv2d(num_chs[-2], num_chs[-3], 3, 1, 1),
            torchvision.ops.DeformConv2d(num_chs[-3], num_chs[-3], 3, 1, 1)
        ])
        self.fusion_block = nn.ModuleList([
            conv_resblock_one(num_chs[-2], num_chs[-1]),
            conv_resblock_one(num_chs[-2] + num_chs[-3], num_chs[-2]),
            conv_resblock_one(num_chs[-2] + num_chs[-3], num_chs[-3])
        ])

    def forward(self, F_0s, F_1s, E_0):
        Fe_0s = self.event_encoder(E_0)
        
        # ------0------
        feat_t_in = torch.cat((F_0s[-1], Fe_0s[-1], F_1s[-1]), 1)
        off_0, m_0 = self.offset_estimator[0](feat_t_in)
        F0_0_ = self.deform_conv[0](F_0s[-1], off_0, m_0)
        off_0_up = resize_2d(off_0, F_0s[-2])
        F0_0_out = self.fusion_block[0](F0_0_)

        # ------1-------
        feat_t_in = torch.cat((F_0s[-2], Fe_0s[-2], F_1s[-2]), 1)
        off_1, m_1 = self.offset_estimator[1](feat_t_in, off_0_up)
        F0_1_ = self.deform_conv[1](F_0s[-2], off_1, m_1)
        off_1_up = resize_2d(off_1, F_0s[-3])
        F0_0_up = resize_2d(F0_0_, F_0s[-2])
        F0_1_ = self.fusion_block[1](torch.cat((F0_1_, F0_0_up), 1))

        # ------2-------
        feat_t_in = torch.cat((F_0s[-3], Fe_0s[-3], F_1s[-3]), 1)
        off_2, m_2 = self.offset_estimator[2](feat_t_in, off_1_up)
        F0_2_ = self.deform_conv[2](F_0s[-3], off_2, m_2)
        F0_1_up = resize_2d(F0_1_, F_0s[-3])
        F0_2_ = self.fusion_block[2](torch.cat((F0_2_, F0_1_up), 1))

        return F0_2_, F0_1_, F0_0_out


class EncoderRef(nn.Module):
    def __init__(self, num_chs):
        super(EncoderRef, self).__init__()
        self.conv1 = conv_resblock_one(num_chs[0], num_chs[1], stride=1)
        self.conv2 = conv_resblock_one(num_chs[1], num_chs[2], stride=2)
        self.conv3 = conv_resblock_one(num_chs[2], num_chs[3], stride=2)
        self.conv4 = conv_resblock_one(num_chs[3], num_chs[4], stride=2)

    def forward(self, image):
        x = self.conv1(image)
        f1 = self.conv2(x)
        f2 = self.conv3(f1)
        f3 = self.conv4(f2)
        return f1, f2, f3


class frame_encoder(nn.Module):
    def __init__(self, in_dims, nf):
        super(frame_encoder, self).__init__()
        self.conv0 = conv3x3_leaky_relu(in_dims, nf)
        self.conv1 = conv_resblock_two(nf, nf)
        self.conv2 = conv_resblock_two(nf, 2 * nf, stride=2)
        self.conv3 = conv_resblock_two(2 * nf, 4 * nf, stride=2)

    def forward(self, x):
        x_ = self.conv0(x)
        f1 = self.conv1(x_)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return [f1, f2, f3]


class event_encoder(nn.Module):
    def __init__(self, in_dims, nf):
        super(event_encoder, self).__init__()
        self.conv0 = conv3x3_leaky_relu(in_dims, nf)
        self.conv1 = conv_resblock_two(nf, nf)
        self.conv2 = conv_resblock_two(nf, 2 * nf, stride=2)
        self.conv3 = conv_resblock_two(2 * nf, 4 * nf, stride=2)

    def forward(self, x):
        x_ = self.conv0(x)
        f1 = self.conv1(x_)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return [f1, f2, f3]


class ref_encoder(nn.Module):
    def __init__(self, in_dims, nf):
        super(ref_encoder, self).__init__()
        self.conv0 = conv3x3_leaky_relu(in_dims, nf)
        self.conv1 = conv_resblock_two(nf, nf)
        self.conv2 = conv_resblock_two(nf, 2 * nf, stride=2)
        self.conv3 = conv_resblock_two(2 * nf, 4 * nf, stride=2)

    def forward(self, x):
        x_ = self.conv0(x)
        f1 = self.conv1(x_)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return [f1, f2, f3]
