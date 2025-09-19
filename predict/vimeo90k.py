import cv2
import lpips
import numpy as np
import os
import torch
import glob
from PIL import Image
from predict.pred_utils import GroupedMetricLogger, flolpips_, get_voxel_and_mask, lpips_, psnr_, ssim_, dists_
from util.event import EventSequence
import torchvision.transforms as transforms

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
        
to_tensor = transforms.Compose([
    transforms.ToTensor()
])

def predict_vimeo90k(model, num_bins, device, save_path, isSave=False, isTestPer=False, 
                  saveIndexRange=None):
    data_root = '/DATASSD/vimeo_triplet'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    image_root = os.path.join(data_root, 'sequences')
    test_fn = os.path.join(data_root, 'tri_testlist.txt')
    with open(test_fn, 'r') as f:
        testlist = f.read().splitlines()
    
    loss_logger = GroupedMetricLogger()
    for index in range(len(testlist)):
        if index != "":
            img_folder = os.path.join(image_root, testlist[index])
            event_folder = os.path.join(img_folder, 'events')
            names = os.listdir()
            img_paths = sorted(glob.glob(os.path.join(img_folder, "im[0-9].png")))
            
            img0 = Image.open(img_paths[0])
            img1 = Image.open(img_paths[2])
            gt = Image.open(img_paths[1])
            w, h = gt.size
            pure_gt = gt.copy()
            pure_gt = to_tensor(pure_gt).unsqueeze(0).to(device, non_blocking=True)
            
            img0 = normalize(img0).unsqueeze(0).to(device, non_blocking=True)
            gt = normalize(gt).unsqueeze(0).to(device, non_blocking=True)
            img1 = normalize(img1).unsqueeze(0).to(device, non_blocking=True)
            

            timestamps_path = os.path.join(img_folder, 'timestamps.txt')
            with open(timestamps_path, 'r') as f:
                timestamps = f.read().splitlines()

            event_ind0 = 0
            event_ind1 = timestamps.index(str(0.3333333333333333))
            event_ind2 = timestamps.index(str(0.6666666666666666))
            before_event_paths = [os.path.join(event_folder, str(i).zfill(10) + ".npz") for i in
                                    range(event_ind0, event_ind1)]
            after_event_paths = [os.path.join(event_folder, str(i).zfill(10) + ".npz") for i in
                                    range(event_ind1, len(os.listdir(event_folder)))]
            events_0t = EventSequence.from_npz_files(before_event_paths, h, w)
            events_t1 = EventSequence.from_npz_files(after_event_paths, h, w)
            event_voxel, mask = get_voxel_and_mask(num_bins, events_0t, events_t1, h, w)
            
            batch = {'img0': img0,
                     'gt': gt,
                     'pure_gt': pure_gt,
                     'img1': img1,
                     'e0t': event_voxel[:, :8],
                     'e1t': event_voxel[:, 16:]}
            
            with torch.no_grad():
                pred = model.inference(batch)
            
            flolpips = flolpips_(img0, img1, pred, pure_gt)
            dists = dists_(pure_gt, pred)
            pred_out = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            pure_gt = (pure_gt[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            psnr = psnr_(pure_gt, pred_out)
            ssim = ssim_(pure_gt, pred_out)
            lpips = lpips_(pure_gt, pred_out)
            
            loss_logger.update("all", psnr, ssim, lpips, flolpips, dists)
            save_name = os.path.join(save_path, str(index).zfill(6) + ".png")
            if isTestPer:
                print(psnr, "  ", ssim, " ", lpips, " ", flolpips)
                
            if isSave:
                if isinstance(saveIndexRange, list):
                    if index >= saveIndexRange[0] and index <= saveIndexRange[1]:
                        cv2.imwrite(save_name, pred_out)
                else:
                    cv2.imwrite(save_name, pred_out)
    loss_logger.print_summary("all")