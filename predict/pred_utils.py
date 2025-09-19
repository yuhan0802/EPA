import os
import io
import cv2
import lpips
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from util.event import EventSequence
from util.utils_func import events_to_channels, process_mask
from PIL import Image
import torch
from util.voxelization import to_voxel_grid
import warnings
import torchvision.transforms as transforms
from flolpipsloss.flolpips import Flolpips
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
        
to_tensor = transforms.Compose([
    transforms.ToTensor()
])

device = torch.device("cuda")
warnings.filterwarnings('ignore')
loss_lpips = lpips.LPIPS(net='alex').to(device)
loss_flolpips = Flolpips().to(device)
loss_dists = DeepImageStructureAndTextureSimilarity().to(device)

def psnr_(gt, pred):
    return peak_signal_noise_ratio(gt, pred)


def ssim_(gt, pred):
    multichannel = len(gt.shape) == 3 and gt.shape[2] == 3
    return structural_similarity(gt, pred, data_range=gt.max() - gt.min(), multichannel=multichannel,
                                 gaussian_weights=True, channel_axis=2)

def lpips_(gt, pred):
    return loss_lpips(gt, pred).item()

def flolpips_(img0, img1, pure_gt, pred):
    flolpips = loss_flolpips.forward(img0, img1, pred, pure_gt)
    return flolpips.item()

def dists_(gt, img):
    return loss_dists(gt, img).item()
    
def looking_for_event_index_by_timestamp(path, bins, img_index0, gt_index, img_index1, h, w, frames, real=False, bsergb=False):
    event_folder = os.path.join(path, 'events')
    if not real:
        timestamps_path = os.path.join(path, 'timestamps.txt')
        with open(timestamps_path, 'r') as f:
            timestamps = f.read().splitlines()

        start_ind = timestamps.index(str(img_index0 / frames))
        gt_ind = timestamps.index(str(gt_index / frames))
        end_ind = timestamps.index(str(img_index1 / frames))

        before_event_paths = [os.path.join(event_folder, str(i).zfill(10) + ".npz") for i in
                              range(start_ind, gt_ind)]
        after_event_paths = [os.path.join(event_folder, str(i).zfill(10) + ".npz") for i in
                             range(gt_ind, end_ind)]
    else:
        before_event_paths = [os.path.join(event_folder, str(i).zfill(6) + ".npz") for i in
                              range(img_index0, gt_index)]
        after_event_paths = [os.path.join(event_folder, str(i).zfill(6) + ".npz") for i in
                             range(gt_index, img_index1)]
    try:
        events_0t = EventSequence.from_npz_files(before_event_paths, h, w, bsergb=bsergb, size=[0, 0, h, w])
    except:
        events_0t = None
    try:
        events_t1 = EventSequence.from_npz_files(after_event_paths, h, w, bsergb=bsergb, size=[0, 0, h, w])
    except:
        events_t1 = None
    if events_0t is None:
        event_0t_voxel = torch.zeros([bins, h, w])
        ec_0t = torch.zeros([2, h, w])
    else: 
        event_0t_voxel = to_voxel_grid(events_0t, bins)
        ec_0t = events_to_channels(events_0t)
    if events_t1 is None:
        event_t1_voxel = torch.zeros([bins, h, w])
        event_1t_voxel = torch.zeros([bins, h, w])
        ec_t1 = torch.zeros([2, h, w])
    else:
        event_t1_voxel = to_voxel_grid(events_t1, bins)
        ec_t1 = events_to_channels(events_t1)
        event_1t = events_t1.reverse()
        event_1t_voxel = to_voxel_grid(event_1t, bins)
    event_cnt = torch.cat((ec_0t, ec_t1), 0)
    event_cnt = process_mask(event_cnt)
    event_voxel = torch.cat((event_0t_voxel, event_t1_voxel, event_1t_voxel), 0).unsqueeze(0).to(device, non_blocking=True)
    mask = event_cnt.unsqueeze(0).to(device, non_blocking=True)
    return event_voxel, mask


def get_voxel_and_mask(num_bins, events_0t, events_t1, h, w):
    if events_0t is None:
        event_0t_voxel = torch.zeros([num_bins, h, w])
        ec_0t = torch.zeros([2, h, w])
    else: 
        event_0t_voxel = to_voxel_grid(events_0t, num_bins)
        ec_0t = events_to_channels(events_0t)
    if events_t1 is None:
        event_t1_voxel = torch.zeros([num_bins, h, w])
        event_1t_voxel = torch.zeros([num_bins, h, w])
        ec_t1 = torch.zeros([2, h, w])
    else:
        event_t1_voxel = to_voxel_grid(events_t1, num_bins)
        ec_t1 = events_to_channels(events_t1)
        event_1t = events_t1.reverse()
        event_1t_voxel = to_voxel_grid(event_1t, num_bins)
    event_cnt = torch.cat((ec_0t, ec_t1), 0)
    event_cnt = process_mask(event_cnt)
    event_voxel = torch.cat((event_0t_voxel, event_t1_voxel, event_1t_voxel), 0).unsqueeze(0).to(device, non_blocking=True)
    mask = event_cnt.unsqueeze(0).to(device, non_blocking=True)
    return event_voxel, mask


def get_imgs(img_paths, multi, index):
    img0 = Image.open(img_paths[index])
    img1 = Image.open(img_paths[index + multi + 1])
    img0 = normalize(img0).unsqueeze(0).to(device, non_blocking=True)
    img1 = normalize(img1).unsqueeze(0).to(device, non_blocking=True)
    return img0, img1

# 获取文件夹下所有场景名
def get_scenes(path):
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


class MetricLogger:
    def __init__(self):
        self.metrics = {
            'psnr': [],
            'ssim': [],
            'lpips': [],
            'flolpips': []
        }

    def update(self, psnr, ssim, lpips, flolpips, dists):
        self.metrics['psnr'].append(psnr)
        self.metrics['ssim'].append(ssim)
        self.metrics['lpips'].append(lpips)
        self.metrics['flolpips'].append(flolpips)
        self.metrics['dists'].append(dists)

    def mean(self):
        return {k: np.mean(v) for k, v in self.metrics.items()}

    def print_summary(self):
        mean_metrics = self.mean()
        print(f"PSNR: {mean_metrics['psnr']:.4f}")
        print(f"SSIM: {mean_metrics['ssim']:.4f}")
        print(f"LPIPS: {mean_metrics['lpips']:.4f}")
        print(f"FloLPIPS: {mean_metrics['flolpips']:.4f}")


class GroupedMetricLogger:
    def __init__(self):
        self.groups = {}

    def update(self, group_name, psnr, ssim, lpips, flolpips, dists):
        if group_name not in self.groups:
            self.groups[group_name] = {'psnr': [], 'ssim': [], 'lpips': [], 'flolpips': [], 'dists': []}
        self.groups[group_name]['psnr'].append(psnr)
        self.groups[group_name]['ssim'].append(ssim)
        self.groups[group_name]['lpips'].append(lpips)
        self.groups[group_name]['flolpips'].append(flolpips)
        self.groups[group_name]['dists'].append(dists)

    def mean(self, group_name):
        if group_name not in self.groups:
            raise ValueError(f"Group {group_name} not found!")
        return {k: np.mean(v) for k, v in self.groups[group_name].items()}

    def print_summary(self, group_name):
        mean_metrics = self.mean(group_name)
        print(f"[{group_name}] PSNR: {mean_metrics['psnr']:.4f}  SSIM: {mean_metrics['ssim']:.4f}  LPIPS: {mean_metrics['lpips']:.4f}  FloLPIPS: {mean_metrics['flolpips']:.4f}  DISTS: {mean_metrics['dists']:.4f}")

    def print_all_summaries(self):
        for group_name in self.groups.keys():
            self.print_summary(group_name)
