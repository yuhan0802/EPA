import cv2
import numpy as np
import os
import torch
from numba import jit
from PIL import Image
import torchvision.transforms as transforms
from predict.pred_utils import GroupedMetricLogger, dists_, flolpips_, get_imgs, looking_for_event_index_by_timestamp, lpips_, psnr_, ssim_

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
        
to_tensor = transforms.Compose([
    transforms.ToTensor()
])


def predict_aid(model, num_bins, device, save_path, multis=[7, 15], isSave=False, isTestPer=False, 
                  saveSpecificScene=None):
    logger = GroupedMetricLogger()
    print("Start test EventAid-F!")
    test_path = '/DATASSD1/EventAid-F'
    for multi in multis:
        for scene in os.listdir(test_path):
            if scene != '':
                val_folder = os.path.join(save_path, scene + '_' + str(multi))
                img_folder = os.path.join(test_path, scene, 'gt')
                img_names = os.listdir(img_folder)
                img_names.sort()
                os.makedirs(val_folder, exist_ok=True)
                img_paths = [os.path.join(img_folder, i) for i in img_names if i.endswith("png") or i.endswith("jpg")]
                img_paths.sort()
                h, w, _ = cv2.imread(img_paths[0]).shape
                index = 0
                while (index + multi + 2) < len(img_names):
                    img0, img1 = get_imgs(img_paths, multi, index)
                    for i in range(multi):
                        gt = Image.open(img_paths[index + i + 1])
                        if index + i + 1 < 332:
                            continue
                        w, h = gt.size
                        pure_gt = gt.copy()
                        pure_gt = to_tensor(pure_gt).unsqueeze(0).to(device, non_blocking=True)
                        gt = normalize(gt).unsqueeze(0).to(device, non_blocking=True)
                        img_index0, gt_index, img_index1 = index, index + i + 1, index + multi + 1
                        
                        event_voxel = looking_for_event_index_by_timestamp(
                                os.path.join(test_path, scene), num_bins,
                                img_index0, gt_index,
                                img_index1, h, w, 0, real=True, aid=True)
                        
                        event_voxel = event_voxel.to(device, non_blocking=True)
                        
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
                        
                        logger.update(f"{scene}_{multi}", psnr, ssim, lpips, flolpips, dists)
                        logger.update(f"multi_{multi}", psnr, ssim, lpips, flolpips, dists)
                        if isTestPer:
                            print(scene, "\'s ", psnr, "  ", ssim, " ", lpips, " ", flolpips, " ", dists)
                        
                        save_name = os.path.join(val_folder, os.path.basename(img_names[index + i + 1]))
                        if isSave:
                            if isinstance(saveSpecificScene, list):
                                if scene in saveSpecificScene:
                                    img = Image.fromarray(pred_out)
                                    img.save(save_name)
                            else:
                                img = Image.fromarray(pred_out)
                                img.save(save_name)
                    index = index + multi
                logger.print_summary(f"{scene}_{multi}")
        logger.print_summary(f"multi_{multi}")
