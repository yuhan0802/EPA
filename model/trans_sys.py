# from model.patch_attention import DetailsSupCross
from model.patch_attention import DetailsSup
from model.submodules import *
import torch
import torch.nn as nn
from einops import rearrange
import numbers
import torch.nn.functional as F


##################################################
################# Restormer #####################

##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv1 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv2 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv1_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.kv2_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x, attn_kv1, attn_kv2):
        b, c, h, w = x.shape

        q_ = self.q_dwconv(self.q(x))
        kv1 = self.kv1_dwconv(self.kv1(attn_kv1))
        kv2 = self.kv2_dwconv(self.kv2(attn_kv2))
        q1, q2 = q_.chunk(2, dim=1)
        k1, v1 = kv1.chunk(2, dim=1)
        k2, v2 = kv2.chunk(2, dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        attn = (q1 @ k1.transpose(-2, -1)) * self.temperature1
        attn = attn.softmax(dim=-1)
        out1 = (attn @ v1)

        attn = (q2 @ k2.transpose(-2, -1)) * self.temperature2
        attn = attn.softmax(dim=-1)
        out2 = (attn @ v2)

        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = torch.cat((out1, out2), dim=1)
        out = self.project_out(out)
        return out


##########################################################################
class CrossTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(CrossTransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm_kv1 = LayerNorm(dim, LayerNorm_type)
        self.norm_kv2 = LayerNorm(dim, LayerNorm_type)
        self.attn = CrossAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, attn_kv1, attn_kv2):
        x = x + self.attn(self.norm1(x), self.norm_kv1(attn_kv1), self.norm_kv2(attn_kv2))
        x = x + self.ffn(self.norm2(x))
        return x


class CrossTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks):
        super(CrossTransformerLayer, self).__init__()
        self.blocks = nn.ModuleList([CrossTransformerBlock(dim=dim, num_heads=num_heads,
                                                           ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                           LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])

    def forward(self, x, attn_kv=None, attn_kv2=None):
        for blk in self.blocks:
            x = blk(x, attn_kv, attn_kv2)
        return x

    ##########################################################################


## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Self_attention(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Self_attention, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks):
        super(SelfAttentionLayer, self).__init__()
        self.blocks = nn.ModuleList(
            [Self_attention(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                            LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)

    def forward(self, x):
        out = self.deconv(x)
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * 2 * W * 2 * self.in_channel * self.out_channel * 2 * 2
        print("Upsample:{%.2f}" % (flops / 1e9))
        return flops


class Transformer(nn.Module):
    def __init__(self, unit_dim):
        super(Transformer, self).__init__()
        # init qurey networks
        self.init_qurey_net(unit_dim)
        self.init_decoder(unit_dim)
        # last conv
        self.last_conv0 = conv3x3(unit_dim * 4, 3)
        self.last_conv1 = conv3x3(unit_dim * 2, 3)
        self.last_conv2 = conv3x3(unit_dim, 3)
        self.activate = torch.sigmoid

        # 降维
        self.conv0 = conv3x3(unit_dim * 4, unit_dim // 2)
        self.conv1 = conv3x3(unit_dim * 2, unit_dim // 4)

        # 新模块
        self.refine_block0 = DetailsSup(patch_size=8, channels=unit_dim // 2, dim=unit_dim // 2)
        self.refine_block1 = DetailsSup(patch_size=16, channels=unit_dim // 4, dim=unit_dim // 4)

    def init_decoder(self, unit_dim):
        # decoder
        # attention k,v building (event)
        self.build_kv0_syn = conv3x3_leaky_relu(unit_dim * 3, unit_dim * 4)
        self.build_kv1_syn = conv3x3_leaky_relu(int(unit_dim * 1.5), unit_dim * 2)
        self.build_kv2_syn = conv3x3_leaky_relu(int(unit_dim * 0.75), unit_dim)
        # attention k, v building (warping)
        self.build_kv0_warp = conv3x3_leaky_relu(unit_dim * 3, unit_dim * 4)
        self.build_kv1_warp = conv3x3_leaky_relu(int(unit_dim * 1.5), unit_dim * 2)
        self.build_kv2_warp = conv3x3_leaky_relu(int(unit_dim * 0.75), unit_dim)
        # level 1
        self.decoder1_1 = CrossTransformerLayer(dim=unit_dim * 4, num_heads=4, ffn_expansion_factor=2.66, bias=False,
                                                LayerNorm_type='WithBias', num_blocks=2)
        self.decoder1_2 = SelfAttentionLayer(dim=unit_dim * 4, num_heads=4, ffn_expansion_factor=2.66, bias=False,
                                             LayerNorm_type='WithBias', num_blocks=2)
        # level 2
        self.decoder2_1 = CrossTransformerLayer(dim=unit_dim * 2, num_heads=4, ffn_expansion_factor=2.66, bias=False,
                                                LayerNorm_type='WithBias', num_blocks=2)
        self.decoder2_2 = SelfAttentionLayer(dim=unit_dim * 2, num_heads=2, ffn_expansion_factor=2.66, bias=False,
                                             LayerNorm_type='WithBias', num_blocks=2)
        # level 3
        self.decoder3_1 = CrossTransformerLayer(dim=unit_dim, num_heads=4, ffn_expansion_factor=2.66, bias=False,
                                                LayerNorm_type='WithBias', num_blocks=2)
        self.decoder3_2 = SelfAttentionLayer(dim=unit_dim, num_heads=1, ffn_expansion_factor=2.66, bias=False,
                                             LayerNorm_type='WithBias', num_blocks=2)
        # upsample
        self.upsample0 = Upsample(unit_dim * 4, unit_dim * 2)
        self.upsample1 = Upsample(unit_dim * 2, unit_dim)
        # conv after body
        self.conv_after_body0 = conv_resblock_one(4 * unit_dim, 2 * unit_dim)
        self.conv_after_body1 = conv_resblock_one(2 * unit_dim, unit_dim)

    # qurey network
    def init_qurey_net(self, unit_dim):
        # building query
        # stage 1
        self.enc_conv0 = conv3x3_leaky_relu(unit_dim, unit_dim)
        # stage 2
        self.enc_conv1 = conv3x3_leaky_relu(unit_dim, 2 * unit_dim, stride=2)
        # stage 3
        self.enc_conv2 = conv3x3_leaky_relu(2 * unit_dim, 4 * unit_dim, stride=2)

    # query buiding
    def build_qurey(self, event_feature, frame_feature, warped_feature):
        cat_in0 = torch.cat((event_feature[0], frame_feature[0], warped_feature[0]), dim=1)
        Q0 = self.enc_conv0(cat_in0)
        Q1 = self.enc_conv1(Q0)
        Q2 = self.enc_conv2(Q1)
        return [Q0, Q1, Q2]

    def forward_decoder(self, Q_list, ref_feature, frame_feature, event_feature, f_frame0, f_frame1, corr0_list,
                        corr1_list):
        # syntheis kv building
        cat_in0_syn = torch.cat((frame_feature[2], event_feature[2]), dim=1)
        attn_kv0_syn = self.build_kv0_syn(cat_in0_syn)
        cat_in1_syn = torch.cat((frame_feature[1], event_feature[1]), dim=1)
        attn_kv1_syn = self.build_kv1_syn(cat_in1_syn)
        cat_in2_syn = torch.cat((frame_feature[0], event_feature[0]), dim=1)
        attn_kv2_syn = self.build_kv2_syn(cat_in2_syn)
        # warping kv building
        cat_in0_warp = torch.cat((ref_feature[2], event_feature[2]), dim=1)
        attn_kv0_warp = self.build_kv0_warp(cat_in0_warp)
        cat_in1_warp = torch.cat((ref_feature[1], event_feature[1]), dim=1)
        attn_kv1_warp = self.build_kv1_warp(cat_in1_warp)
        cat_in2_warp = torch.cat((ref_feature[0], event_feature[0]), dim=1)
        attn_kv2_warp = self.build_kv2_warp(cat_in2_warp)
        # out 0
        _Q0 = Q_list[2]
        out0 = self.decoder1_1(_Q0, attn_kv0_syn, attn_kv0_warp)
        # B, 256, 64 ,64
        out0 = self.decoder1_2(out0)
        # B, 128, 128, 128
        up_out0 = self.upsample0(out0)
        # B, 3, 64, 64
        img0 = self.activate(self.last_conv0(out0))

        # 新模块
        # B, 32, 64, 64
        out0 = self.conv0(out0)
        up_out0 = self.refine_block0(out0, f_frame0[2], f_frame1[2], up_out0, corr0_list[2], corr1_list[2])

        # out 1
        _Q1 = Q_list[1]
        _Q1 = self.conv_after_body0(torch.cat((_Q1, up_out0), dim=1))
        out1 = self.decoder2_1(_Q1, attn_kv1_syn, attn_kv1_warp)
        # B, 128, 128, 128
        out1 = self.decoder2_2(out1)
        # B, 64, 256, 256
        up_out1 = self.upsample1(out1)
        # B, 3, 128, 128
        img1 = self.activate(self.last_conv1(out1))

        # 新模块
        # B, 16, 128, 128
        out1 = self.conv1(out1)
        up_out1 = self.refine_block1(out1, f_frame0[1], f_frame1[1], up_out1, corr0_list[1], corr1_list[1])

        # out2
        _Q2 = Q_list[0]
        _Q2 = self.conv_after_body1(torch.cat((_Q2, up_out1), dim=1))
        out2 = self.decoder3_1(_Q2, attn_kv2_syn, attn_kv2_warp)
        out2 = self.decoder3_2(out2)
        img2 = self.activate(self.last_conv2(out2))

        return img0, img1, img2

    def forward(self, event_feature, frame_feature, ref_feature, f_frame0, f_frame1, corr0_list, corr1_list):
        # f_frame 8, 16, 32
        # forward encoder
        Q_list = self.build_qurey(event_feature, frame_feature, ref_feature)
        # forward decoder
        img0, img1, img2 = self.forward_decoder(Q_list, ref_feature, frame_feature, event_feature, f_frame0, f_frame1,
                                                corr0_list, corr1_list)
        return [img2, img1, img0]