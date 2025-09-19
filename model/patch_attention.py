import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from model.position_encoding import build_position_encoding
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
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


class isAttenChange(object):

    def __init__(self):
        self.H = 0
        self.W = 0
        self.atten_mask = None

    def get_attention_mask(self, H, W, patch_size):
        # 计算图像的行列数
        num_rows = H // patch_size
        num_cols = W // patch_size

        # 计算patch数量
        num_patches = num_rows * num_cols

        # 创建一个大小为(num_patches, num_patches)的注意力矩阵，初始化为0
        attention_mask = torch.zeros(num_patches, num_patches)

        # 对于每个patch
        for i in range(num_patches):
            # 计算当前patch在图像中的行列索引
            row = i // num_cols
            col = i % num_cols

            # 获取当前patch周围一圈的patch的索引，并考虑边界情况
            row_indices = torch.arange(row - 1, row + 2).clamp(0, num_rows - 1)
            col_indices = torch.arange(col - 1, col + 2).clamp(0, num_cols - 1)
            neighbor_indices = col_indices.unsqueeze(0).T + row_indices.unsqueeze(0) * num_cols
            neighbor_indices = neighbor_indices.flatten()

            # 将当前patch及其周围一圈的patch之间的注意力设为1
            attention_mask[i, neighbor_indices] = 1

        attention_mask = attention_mask.masked_fill(attention_mask == 0, float(-100.0)).masked_fill(attention_mask == 1,
                                                                                                    float(0.0))

        return attention_mask

    def judge(self, H, W, patch_size):
        if H != self.H or W != self.W:
            self.atten_mask = self.get_attention_mask(H, W, patch_size)
            self.H = H
            self.W = W
            return self.atten_mask
        else:
            return self.atten_mask


class TransformerBlock(nn.Module):
    def __init__(self, patch_size, dim, channels, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.patch_size = patch_size
        patch_dim = channels * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            # nn.Linear(patch_dim, dim),
        )
        self.to_patch_embedding2 = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = build_position_encoding('sine', dim)
        self.dropout = nn.Dropout(dropout)

        self.head = num_heads
        inner_dim = int(mlp_ratio * dim)
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.img0_to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.img1_to_qv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = Mlp(in_features=dim, hidden_features=inner_dim, out_features=patch_dim, drop=dropout)
        self.mlp2 = Mlp(in_features=dim, hidden_features=inner_dim, out_features=patch_dim, drop=dropout)
        self.isAttenChange = isAttenChange()
        self.out = nn.Conv2d(dim, channels, 3, 1, 1)

    def forward(self, img0, img1, local_or_global='local', mask=None):
        B, C, H, W = img0.shape
        # img1 去 img0 上查相似来补足img1
        # img0 : B, 256, 64, 64
        img0 = self.to_patch_embedding(img0)
        img0 = self.to_patch_embedding2(img0)
        img1 = self.to_patch_embedding(img1)
        img1 = self.to_patch_embedding2(img1)
        b, n, c = img0.shape

        img0 += self.pos_embedding(img0)
        img0 = self.dropout(img0)
        img1 += self.pos_embedding(img1)
        img1 = self.dropout(img1)

        img0_kv = self.img0_to_kv(img0).reshape(b, n, 2, self.head, c // self.head).permute(2, 0, 3, 1, 4)
        img0_k, img0_v = img0_kv[0], img0_kv[1]

        img1_qv = self.img1_to_qv(img1).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        img1_q, img1_v = img1_qv[0], img1_qv[1]
        img1_q = img1_q.reshape(b, n, self.head, c // self.head).permute(0, 2, 1, 3)

        img1_dots = img1_q @ img0_k.transpose(-2, -1).contiguous() * self.scale
        if local_or_global == 'local':
            atten_mask = self.isAttenChange.judge(H, W, self.patch_size).to(device)
            img1_dots = img1_dots + atten_mask.unsqueeze(0).unsqueeze(0)
        elif mask is not None:
            mask = mask.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
            mask = mask.contiguous().view(B, 1, -1, self.patch_size, self.patch_size)
            mask = mask.sum(dim=[3, 4])
            mask[mask > 0] = 1
            mask = mask.unsqueeze(1).repeat(1, self.head, n, 1)
            mask = mask.masked_fill(mask == 0, float(-100.0)).masked_fill(mask == 1, float(0.0))
            img1_dots = img1_dots + mask
        else:
            pass
        
        img1_attn = self.softmax(img1_dots)
        img1_out = (img1_attn @ img0_v).transpose(1, 2).reshape(b, n, c)
        img1_res = self.mlp2(img1_out)
        img1_res = rearrange(img1_res, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size, h=H // self.patch_size)
        return img1_res
    
# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(FeedForward, self).__init__()
#         hidden_features = int(dim * ffn_expansion_factor)
#         self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
#         self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
#                                 groups=hidden_features * 2, bias=bias)
#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

#     def forward(self, x):
#         x = self.project_in(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x1) * x2
#         x = self.project_out(x)
#         return x
    

# class CrossAttention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(CrossAttention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
#         self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.q = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
#         self.kv1 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
#         self.kv2 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
#         self.q_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
#         self.kv1_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
#         self.kv2_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
#         self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

#     def forward(self, x, attn_kv1, attn_kv2, lmask, rmask):
#         b, c, h, w = x.shape
#         lamsk = lamsk.masked_fill(lamsk == 0, float(-100.0)).masked_fill(lamsk == 1, float(0.0))
#         rmask = rmask.masked_fill(rmask == 0, float(-100.0)).masked_fill(rmask == 1, float(0.0))

#         q_ = self.q_dwconv(self.q(x))
#         kv1 = self.kv1_dwconv(self.kv1(attn_kv1))
#         kv2 = self.kv2_dwconv(self.kv2(attn_kv2))
#         q1, q2 = q_.chunk(2, dim=1)
#         k1, v1 = kv1.chunk(2, dim=1)
#         k2, v2 = kv2.chunk(2, dim=1)

#         q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

#         q1 = torch.nn.functional.normalize(q1, dim=-1)
#         q2 = torch.nn.functional.normalize(q2, dim=-1)
#         k1 = torch.nn.functional.normalize(k1, dim=-1)
#         k2 = torch.nn.functional.normalize(k2, dim=-1)

#         attn = (q1 @ k1.transpose(-2, -1)) * self.temperature1 + lmask
#         attn = attn.softmax(dim=-1)
#         out1 = (attn @ v1)

#         attn = (q2 @ k2.transpose(-2, -1)) * self.temperature2 + rmask
#         attn = attn.softmax(dim=-1)
#         out2 = (attn @ v2)

#         out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#         out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#         out = torch.cat((out1, out2), dim=1)
#         out = self.project_out(out)
#         return out
    
# class CrossTransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
#         super(CrossTransformerBlock, self).__init__()
#         self.norm1 = LayerNorm(dim, LayerNorm_type)
#         self.norm_kv1 = LayerNorm(dim, LayerNorm_type)
#         self.norm_kv2 = LayerNorm(dim, LayerNorm_type)
#         self.attn = CrossAttention(dim, num_heads, bias)
#         self.norm2 = LayerNorm(dim, LayerNorm_type)
#         self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

#     def forward(self, x, attn_kv1, attn_kv2, lmask, rmask):
#         x = x + self.attn(self.norm1(x), self.norm_kv1(attn_kv1), self.norm_kv2(attn_kv2), lmask, rmask)
#         x = x + self.ffn(self.norm2(x))
#         return x
    

# class DetailsSupCross(nn.Module):
#     def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks):
#         super(DetailsSup, self).__init__()
#         self.blocks = nn.ModuleList([CrossTransformerBlock(dim=dim, num_heads=num_heads,
#                                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias,
#                                                            LayerNorm_type=LayerNorm_type) for _ in range(num_blocks)])

#     def forward(self, img0, x, img1, corr0, corr1):
#         for blk in self.blocks:
#             x = blk(x, img0, img1, corr0, corr1)
#         return x





class DetailsSup(nn.Module):
    def __init__(self, patch_size, dim, channels, heads=8):
        super(DetailsSup, self).__init__()
        self.transformer_block = TransformerBlock(patch_size=patch_size, dim=dim, channels=channels, num_heads=heads)
        self.upsample = Upsample(channels * 2, channels)
        self.fusion_block = nn.Sequential(
            nn.Conv2d(channels * 5, channels, 3, 1, 1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(channels, channels * 4, 3, 1, 1)
        )

    def forward(self, imgt, img0, img1, f_up, corr0, corr1, local_or_global='global'):
        imgt_from_img0 = imgt + self.transformer_block(img0, imgt, local_or_global, corr0)
        imgt_from_img1 = imgt + self.transformer_block(img1, imgt, local_or_global, corr1)
        imgt = self.upsample(torch.cat((imgt_from_img0, imgt_from_img1), 1))
        out = self.fusion_block(torch.cat((imgt, f_up), 1))
        return out


class DetailsSupLayer(nn.Module):
    def __init__(self, patch_size, dim, channels, heads=4, num_blocks=1):
        super(DetailsSupLayer, self).__init__()
        self.blocks = nn.ModuleList([TransformerBlock(patch_size, dim, channels, heads) for _ in range(num_blocks)])

    def forward(self, x, img0, corr):
        for blk in self.blocks:
            x = x + blk(img0, x, mask=corr)
        return x


if __name__ == '__main__':
    unit_dim = 64
    net = TransformerBlock(patch_size=16, channels=unit_dim * 4, dim=unit_dim // 2, num_heads=8)
    summary(net)
