from types import SimpleNamespace

import numpy as np
import torch
import random
import scipy.io as scio
import torch.nn.functional as F
from torch import nn
import cv2
import io
from PIL import Image
from util.event import EventSequence


def binary_search_numpy_array(t, l, r, x, side='left'):
    """
    Binary search sorted numpy array
    """
    if r is None:
        r = len(t) - 1
    while l <= r:
        mid = l + (r - l) // 2
        midval = t[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r


def index_patch(patch, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    # print(points.shape, idx.shape)
    device = patch.device
    B = patch.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    # print(B, view_shape, repeat_shape)
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    # print(batch_indices.shape, idx.shape, patch.shape)
    new_patch = patch[batch_indices, idx, :]
    return new_patch


def events_to_image(xs, ys, ps, sensor_size):
    """
    Accumulate events into an image.
    """

    device = xs.device
    img_size = list(sensor_size)
    img = torch.zeros(img_size).to(device)

    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    img.index_put_((ys, xs), ps, accumulate=True)

    return img


def events_to_channels(event: EventSequence):
    """
    Generate a two-channel event image containing event counters.
    """
    xs = event._features_torch[:, 0]
    ys = event._features_torch[:, 1]
    ps = event._features_torch[:, 3]
    sensor_size = [event._image_height, event._image_width]
    assert len(xs) == len(ys) and len(ys) == len(ps)

    mask_pos = ps.clone()
    mask_neg = ps.clone()
    mask_pos[ps < 0] = 0
    mask_neg[ps > 0] = 0

    pos_cnt = events_to_image(xs, ys, ps * mask_pos, sensor_size=sensor_size)
    neg_cnt = events_to_image(xs, ys, ps * mask_neg, sensor_size=sensor_size)

    return torch.stack([pos_cnt, neg_cnt])


def custom_erode(mask, kernel_size=3):
    # 创建一个卷积核，中心为0，其余为1
    kernel = torch.ones((1, 1, kernel_size, kernel_size))
    kernel[0, 0, kernel_size // 2, kernel_size // 2] = 0

    # 计算与中心像素不同的邻居数量
    neighbors_diff = F.conv2d(mask.float(), kernel, padding=kernel_size // 2)

    # 根据邻居像素值更新中心像素
    # change_to_1 = (mask == 0) & (neighbors_diff == (kernel_size * kernel_size - 1))
    # change_to_0 = (mask == 1) & (neighbors_diff == 0)

    change_to_1 = (mask == 0) & (neighbors_diff >= 5)
    change_to_0 = (mask == 1) & (neighbors_diff <= 1)

    mask[change_to_1] = 1
    mask[change_to_0] = 0

    return mask


def dilate(mask, kernel_size=3):
    # 创建一个卷积核
    kernel = torch.ones((1, 1, kernel_size, kernel_size))

    # 膨胀操作
    dilated = F.conv2d(mask.float(), kernel, padding=kernel_size // 2)
    dilated = (dilated > 0).float()

    return dilated


def process_mask(mask, erode_kernel_size=3, dilate_kernel_size=7):
    # 将掩模进行叠加，以得到一个单一的二值图像
    mask = torch.sum(mask, dim=0, keepdim=True)
    mask[mask > 0] = 1

    # 先进行自定义腐蚀操作
    eroded_mask = custom_erode(mask.clone(), erode_kernel_size)

    # 接下来进行膨胀操作
    dilated_mask = dilate(eroded_mask, dilate_kernel_size)

    return dilated_mask
    # return eroded_mask


def make_different_map(img1, img2):
    difference_map = img1 - img2  # [B, 3, H, W], difference map on IB
    difference_map = torch.sum(torch.abs(difference_map), dim=1, keepdim=True)  # [B, 1, H, W]
    return difference_map


# Function to divide the difference map into patches
def divide_into_patches(difference_map, patch_size=32):
    B, C, H, W = difference_map.shape
    patches = difference_map.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)  # [B, C, num_patches, patch_size, patch_size]
    return patches


# 计算每个patch内值的总和
def calculate_patch_sums(patches):
    patch_sums = patches.sum(dim=[3, 4])  # [B, C, num_patches]
    return patch_sums


# 基于总和修改patch
def modify_patches_based_on_sums(patches, patch_sums, sample_factor):
    B, C, num_patches, patch_size, _ = patches.shape
    patch_sums_sorted, indices = torch.sort(patch_sums.view(B, -1), descending=True)
    threshold_index = num_patches // sample_factor
    top_patches_mask = torch.zeros_like(patch_sums.view(B, -1), dtype=torch.bool)
    top_patches_mask[torch.arange(B).unsqueeze(1), indices[:, :threshold_index]] = True

    top_patches_mask = top_patches_mask.view(B, C, num_patches)
    patches = patches.view(B, C, num_patches, patch_size, patch_size)
    patches[~top_patches_mask] = 0
    patches[top_patches_mask] = 1

    return patches.view(B, C, num_patches, patch_size, patch_size)


# 计算patch内是否有1，如果有1则整个patch为1
def modify_patches_based_on_one_or_zero(patches, patch_sums):
    B, C, num_patches, patch_size, _ = patches.shape
    patches_mask = torch.zeros_like(patch_sums.view(B, -1), dtype=torch.bool)
    patches_mask[patch_sums.squeeze(1) > 0] = True

    patches_mask = patches_mask.view(B, C, num_patches)
    patches = patches.view(B, C, num_patches, patch_size, patch_size)
    patches[~patches_mask] = 0
    patches[patches_mask] = 1

    return patches.view(B, C, num_patches, patch_size, patch_size)


# Reconstruct the modified difference map
def reconstruct_from_patches(patches, image_size):
    B, C, num_patches, patch_size, _ = patches.shape
    num_patches_per_row = image_size[1] // patch_size
    num_patches_per_col = image_size[0] // patch_size

    patches = patches.view(B, C, num_patches_per_col, num_patches_per_row, patch_size, patch_size)
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    reconstructed = patches.view(B, C, image_size[0], image_size[1])
    return reconstructed


def pyramid_patch_mask(img0, img1, event_mask, patch_size=32, sample_factor=4):
    difference_map = make_different_map(img0, img1)
    patches = divide_into_patches(difference_map, patch_size)
    patch_sums = calculate_patch_sums(patches)
    modified_patches = modify_patches_based_on_sums(patches, patch_sums, sample_factor)
    modified_difference_map = reconstruct_from_patches(modified_patches, difference_map.shape[2:])

    event_mask_patched = generate_event_mask_patch_mask(event_mask, patch_size)
    modified_difference_map = modified_difference_map * event_mask_patched

    dm_0 = modified_difference_map
    dm_1 = down_sample_mask(modified_difference_map)
    dm_2 = down_sample_mask(dm_1)
    return [dm_0, dm_1, dm_2]


def mask_reverse(mask1, mask2):
    # 创建mask1和mask2的副本以避免原地修改
    new_mask1 = mask1.clone()
    new_mask2 = mask2.clone()

    # 反转mask上不同的区域
    swap_condition = new_mask1 != new_mask2
    temp = new_mask1[swap_condition].clone()  # 临时存储需要交换的区域
    new_mask1[swap_condition] = new_mask2[swap_condition]
    new_mask2[swap_condition] = temp

    return new_mask1, new_mask2


def generate_event_mask_patch_mask(event_mask, patch_size=32):
    patches = divide_into_patches(event_mask, patch_size)
    patch_sums = calculate_patch_sums(patches)
    modified_patches = modify_patches_based_on_one_or_zero(patches, patch_sums)
    modified_event_patches_map = reconstruct_from_patches(modified_patches, event_mask.shape[2:])
    return modified_event_patches_map


def pyramid_Img(Img):
    img_pyr = [Img]
    for i in range(1, 3):
        img_pyr.append(F.interpolate(Img, scale_factor=0.5 ** i, mode='bilinear'))
    return img_pyr


def shrink_mask(event_mask, patch_size):
    mask = generate_event_mask_patch_mask(event_mask, patch_size)
    b, _, height, width = mask.size()
    new_patch_size = int(patch_size / 2)

    patches = mask.view(b, 1, height // patch_size, patch_size, width // patch_size, patch_size)
    patches = patches.permute(0, 1, 2, 4, 3, 5).reshape(b, 1, -1, patch_size, patch_size)

    pooled_patches, _ = torch.max(patches.view(b, 1, -1, 2, new_patch_size, 2, new_patch_size), dim=3, keepdim=True)
    pooled_patches, _ = torch.max(pooled_patches, dim=5, keepdim=True)

    new_height = height // 2
    new_width = width // 2
    new_mask = pooled_patches.view(b, 1, new_height, new_width)

    return new_mask


def down_sample_mask(mask):
    downsampled_tensor = F.avg_pool2d(mask, kernel_size=2)
    downsampled_tensor = torch.round(downsampled_tensor)
    return downsampled_tensor


def convert_dict_to_namespace(d):
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = convert_dict_to_namespace(value)
        return SimpleNamespace(**d)
    else:
        return d


def get_z(heat: float, img_size: tuple, batch: int, device: str):
    def calc_z_shapes(img_size, n_levels):
        h, w = img_size
        z_shapes = []
        channel = 3

        for _ in range(n_levels - 1):
            h //= 2
            w //= 2
            channel *= 2
            z_shapes.append((channel, h, w))
        h //= 2
        w //= 2
        z_shapes.append((channel * 4, h, w))
        return z_shapes

    z_list = []
    z_shapes = calc_z_shapes(img_size, 3)
    for z in z_shapes:
        z_new = torch.randn(batch, *z, device=device) * heat
        z_list.append(z_new)
    return z_list


def Img_pyramid(Img):
    img_pyr = []
    for i in range(1, 4):
        img_pyr.append(F.interpolate(Img, scale_factor=0.5 ** i, mode='bilinear'))
    return img_pyr


class DINOVitFeatureExtractor(nn.Module):
    def __init__(self, dino_model, layers=[0, 5, 11]):
        super().__init__()
        self.dino = dino_model
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False

        self.layers = layers

    def forward(self, x):
        x = self.dino.patch_embed(x)
        B, _, C = x.shape

        cls_token = self.dino.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.dino.pos_embed[:, :x.size(1), :]

        features = []

        for i, blk in enumerate(self.dino.blocks):
            x = blk(x)
            if i in self.layers:
                patch_tokens = x.clone()[:, 1:, :]
                B, N, C = patch_tokens.shape
                H = W = int(N ** 0.5)
                fea = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
                features.append(fea)

        return features


class DINOResNetFeatureExtractor(nn.Module):
    def __init__(self, dino_resnet):
        super().__init__()
        self.model = dino_resnet
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.stem = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
        )
        self.layer1 = self.model.layer1  # shallow
        self.layer2 = self.model.layer2  # middle
        self.layer3 = self.model.layer3  # deeper middle
        self.layer4 = self.model.layer4  # deep

    def forward(self, x):
        feat1 = self.stem(x)  # low
        feat = self.model.maxpool(feat1)
        feat2 = self.layer1(feat)  # middle
        feat3 = self.layer2(feat2)  # deep
        return feat1, feat2, feat3


def add_jpeg_artifact_memory(img: np.ndarray, quality=10):
    # jpeg 伪影
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)

    compressed_img_pil = Image.open(buffer)

    compressed_img = cv2.cvtColor(np.array(compressed_img_pil), cv2.COLOR_RGB2BGR)
    return compressed_img

def add_block_artifact(img, block_size=5):
    # 马赛克 低分辨率
    h, w, _ = img.shape
    img_copy = img.copy()
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = img[y:y+block_size, x:x+block_size]
            avg_color = block.mean(axis=(0, 1))
            img_copy[y:y+block_size, x:x+block_size] = avg_color
    return img_copy