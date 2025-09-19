import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.fft
from math import exp
from util.iwe import get_interpolation, interpolate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EPE(nn.Module):
    def __init__(self):
        super(EPE, self).__init__()

    def forward(self, flow, gt, loss_mask):
        loss_map = (flow - gt.detach()) ** 2
        loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
        return (loss_map * loss_mask)


class Ternary(nn.Module):
    def __init__(self):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(device)

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf ** 2)
        return transf_norm

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, img0, img1):
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)


class SOBEL(nn.Module):
    def __init__(self):
        super(SOBEL, self).__init__()
        self.kernelX = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]).float()
        self.kernelY = self.kernelX.clone().T
        self.kernelX = self.kernelX.unsqueeze(0).unsqueeze(0).to(device)
        self.kernelY = self.kernelY.unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, pred, gt):
        N, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        img_stack = torch.cat(
            [pred.reshape(N * C, 1, H, W), gt.reshape(N * C, 1, H, W)], 0)
        sobel_stack_x = F.conv2d(img_stack, self.kernelX, padding=1)
        sobel_stack_y = F.conv2d(img_stack, self.kernelY, padding=1)
        pred_X, gt_X = sobel_stack_x[:N * C], sobel_stack_x[N * C:]
        pred_Y, gt_Y = sobel_stack_y[:N * C], sobel_stack_y[N * C:]

        L1X, L1Y = torch.abs(pred_X - gt_X), torch.abs(pred_Y - gt_Y)
        loss = (L1X + L1Y)
        return loss


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, rank=0):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        pretrained = True
        self.vgg_pretrained_features = models.vgg19(pretrained=pretrained).features.cuda()
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, Y, indices=None):
        X = self.normalize(X)
        Y = self.normalize(Y)
        indices = [2, 7, 12, 21, 30]
        weights = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10 / 1.5]
        k = 0
        loss = 0
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            Y = self.vgg_pretrained_features[i](Y)
            if (i + 1) in indices:
                loss += weights[k] * (X - Y.detach()).abs().mean() * 0.1
                k += 1
        return loss


class EventWarping(nn.Module):
    """
    Contrast maximization loss, as described in Section 3.2 of the paper 'Unsupervised Event-based Learning
    of Optical Flow, Depth, and Egomotion', Zhu et al., CVPR'19.
    The contrast maximization loss is the minimization of the per-pixel and per-polarity image of averaged
    timestamps of the input events after they have been compensated for their motion using the estimated
    optical flow. This minimization is performed in a forward and in a backward fashion to prevent scaling
    issues during backpropagation.
    """

    def __init__(self, device):
        super(EventWarping, self).__init__()
        self.res = []
        self.flow_scaling = []
        self.device = device

    def forward(self, flow_list, event_list, pol_mask):
        """
        :param flow_list: [[batch_size x 2 x H x W]] list of optical flow maps
        :param event_list: [batch_size x N x 4] input events (y, x, ts, p) (t,y,x,p)
        :param pol_mask: [batch_size x N x 2] per-polarity binary mask of the input events
        """

        _, _, H, W = flow_list.shape
        self.res = [H, W]
        self.flow_scaling = max(self.res)
        # split input
        pol_mask = torch.cat([pol_mask for i in range(4)], dim=1)
        ts_list = torch.cat([event_list[:, :, 0:1] for i in range(4)], dim=1)

        # flow vector per input event
        flow_idx = event_list[:, :, 1:3].clone()
        flow_idx[:, :, 0] *= self.res[1]  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2)

        loss = 0
        for flow in [flow_list]:
            # get flow for every event in the list
            flow = flow.view(flow.shape[0], 2, -1)
            event_flowy = torch.gather(flow[:, 1, :], 1, flow_idx.long())  # vertical component
            event_flowx = torch.gather(flow[:, 0, :], 1, flow_idx.long())  # horizontal component
            event_flowy = event_flowy.view(event_flowy.shape[0], event_flowy.shape[1], 1)
            event_flowx = event_flowx.view(event_flowx.shape[0], event_flowx.shape[1], 1)
            event_flow = torch.cat([event_flowy, event_flowx], dim=2)

            # interpolate forward
            tref = 1
            fw_idx, fw_weights = get_interpolation(event_list, event_flow, tref, self.res, self.flow_scaling)

            # per-polarity image of (forward) warped events
            fw_iwe_pos = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask[:, :, 0:1])
            fw_iwe_neg = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask[:, :, 1:2])

            # image of (forward) warped averaged timestamps
            fw_iwe_pos_ts = interpolate(
                fw_idx.long(), fw_weights * ts_list, self.res, polarity_mask=pol_mask[:, :, 0:1]
            )
            fw_iwe_neg_ts = interpolate(
                fw_idx.long(), fw_weights * ts_list, self.res, polarity_mask=pol_mask[:, :, 1:2]
            )
            fw_iwe_pos_ts /= fw_iwe_pos + 1e-9
            fw_iwe_neg_ts /= fw_iwe_neg + 1e-9

            # interpolate backward
            tref = 0
            bw_idx, bw_weights = get_interpolation(event_list, event_flow, tref, self.res, self.flow_scaling)

            # per-polarity image of (backward) warped events
            bw_iwe_pos = interpolate(bw_idx.long(), bw_weights, self.res, polarity_mask=pol_mask[:, :, 0:1])
            bw_iwe_neg = interpolate(bw_idx.long(), bw_weights, self.res, polarity_mask=pol_mask[:, :, 1:2])

            # image of (backward) warped averaged timestamps
            bw_iwe_pos_ts = interpolate(
                bw_idx.long(), bw_weights * (1 - ts_list), self.res, polarity_mask=pol_mask[:, :, 0:1]
            )
            bw_iwe_neg_ts = interpolate(
                bw_idx.long(), bw_weights * (1 - ts_list), self.res, polarity_mask=pol_mask[:, :, 1:2]
            )
            bw_iwe_pos_ts /= bw_iwe_pos + 1e-9
            bw_iwe_neg_ts /= bw_iwe_neg + 1e-9

            # flow smoothing
            flow = flow.view(flow.shape[0], 2, self.res[0], self.res[1])
            flow_dx = flow[:, :, :-1, :] - flow[:, :, 1:, :]
            flow_dy = flow[:, :, :, :-1] - flow[:, :, :, 1:]
            flow_dx = torch.sqrt(flow_dx ** 2 + 1e-6)  # charbonnier
            flow_dy = torch.sqrt(flow_dy ** 2 + 1e-6)  # charbonnier

            loss += (
                    torch.sum(fw_iwe_pos_ts ** 2)
                    + torch.sum(fw_iwe_neg_ts ** 2)
                    + torch.sum(bw_iwe_pos_ts ** 2)
                    + torch.sum(bw_iwe_neg_ts ** 2)
                    + flow_dx.sum()
                    + flow_dy.sum()
            )

        return loss


class HighFrequencyLoss(nn.Module):
    def __init__(self, mask_radius=15):
        super(HighFrequencyLoss, self).__init__()
        self.mask_radius = mask_radius

    def high_pass_mask(self, size):
        # 创建一个中心为1，其他区域为0的二维遮罩，代表高通滤波器
        mask = torch.zeros(size, size)
        center = size // 2
        mask[center - self.mask_radius:center + self.mask_radius + 1,
        center - self.mask_radius:center + self.mask_radius + 1] = 1
        return 1 - mask

    def complex_mse_loss(self, x, y):
        # 计算复数的实部和虚部的均方误差
        real_mse = nn.functional.mse_loss(x.real, y.real)
        imag_mse = nn.functional.mse_loss(x.imag, y.imag)
        return real_mse + imag_mse

    def forward(self, x, y):
        # 计算图像的傅立叶变换
        fft_x = torch.fft.fftn(x, dim=(-2, -1))
        fft_y = torch.fft.fftn(y, dim=(-2, -1))

        # 应用高通滤波器
        mask = self.high_pass_mask(x.shape[-1]).to(x.device)
        fft_x = fft_x * mask
        fft_y = fft_y * mask

        # 计算损失
        loss = self.complex_mse_loss(fft_x, fft_y)

        return loss
 
 
# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
 
 
# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
 
 
def expand_patches(mask, patch_size):
    B, C, H, W = mask.shape
    patch_H = H // patch_size
    patch_W = W // patch_size

    # reshape和reduce mask到patch表示形式
    small_mask = mask.view(B, C, patch_H, patch_size, patch_W, patch_size).max(dim=5)[0].max(dim=3)[0]

    # 创建膨胀核，这将扩展每个patch的周围一圈
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=mask.device)

    # 应用膨胀
    dilated_small_mask = F.conv2d(small_mask, kernel, padding=1, stride=1)
    dilated_small_mask = (dilated_small_mask > 0).float()  # 二值化处理

    # 将dilated_small_mask上采样回原始大小
    dilated_mask = F.interpolate(dilated_small_mask, size=(H, W), mode='nearest')

    return dilated_mask
 

# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, mask, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
 
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
 
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
        
    # 使用 mask 来决定哪些区域是有效的
    valid_mask = F.conv2d(mask.float(), window, padding=padd, groups=1)
    valid_mask = (valid_mask == 1)  # 确保窗口内全是1
 
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
 
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
 
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
 
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
 
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    
    # 只计算有效区域的平均值
    ssim_map = ssim_map * valid_mask
    
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
 
    if full:
        return ret, cs
    return ret
 
 
 
# Classes to re-use window
class SSIM_with_patch_mask(torch.nn.Module):
    def __init__(self, patch_size, window_size=11, size_average=True, val_range=None):
        super(SSIM_with_patch_mask, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.patch_size = patch_size
 
        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)
 
    def forward(self, img1, img2, mask):
        (_, channel, _, _) = img1.size()
        
        mask = expand_patches(mask, self.patch_size)
        
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
 
        return ssim(img1, img2, mask, window=window, window_size=self.window_size, size_average=self.size_average)

if __name__ == '__main__':
    img0 = torch.zeros(16, 3, 256, 256).float().to(device)
    img1 = torch.tensor(np.random.normal(
        0, 1, (16, 3, 256, 256))).float().to(device)
    ternary_loss = Ternary()
    print(ternary_loss(img0, img1).shape)
