import cv2
import lpips
import numpy as np
import os
import torch

from predict.pred_utils import  get_voxel_and_mask, lpips_, psnr_, ssim_
from util.event import EventSequence
from PIL import Image
from predict.pred_utils import GroupedMetricLogger, dists_, flolpips_, get_imgs, looking_for_event_index_by_timestamp, lpips_, psnr_, ssim_
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


def predict_snufilm(model, num_bins, device, save_path, difficulties=['extreme', 'hard'], isSave=False, isTestPer=False, 
                  saveSpecificScene=None):
    print("Start test SNU-FILM!")
    test_path = '/data/snufilm'
    logger = GroupedMetricLogger()
    for difficulty in difficulties:
        if difficulty == 'hard':
            test_fn = os.path.join(test_path, 'test-hard.txt')
        else:
            test_fn = os.path.join(test_path, 'test-extreme.txt')
        with open(test_fn, 'r') as f:
            test_list = f.read().splitlines()
        val_folder = os.path.join(save_path, difficulty)
        os.makedirs(val_folder, exist_ok=True)
        for index in range(len(test_list)):
            if index != "":
                dataset, scene, name_0, name_gt, name_1 = test_list[index].split(" ")[:5]
                img_folder = os.path.join(test_path, dataset, scene, "imgs")
                
                
                img0 = Image.open(os.path.join(img_folder, name_0))
                img1 = Image.open(os.path.join(img_folder, name_1))
                
                img0 = normalize(img0).unsqueeze(0).to(device, non_blocking=True)
                img1 = normalize(img1).unsqueeze(0).to(device, non_blocking=True)
                
                gt = Image.open(os.path.join(img_folder, name_gt))
                w, h = gt.size
                pure_gt = gt.copy()
                pure_gt = to_tensor(pure_gt).unsqueeze(0).to(device, non_blocking=True)
                gt = normalize(gt).unsqueeze(0).to(device, non_blocking=True)
                
                event_folder = os.path.join(test_path, dataset, scene, "events")
                event_index = [int(i) for i in test_list[index].split(" ")[5:]]
                before_event_paths = [os.path.join(event_folder, str(i).zfill(10) + ".npz") for i in
                                        range(event_index[0], event_index[1])]
                after_event_paths = [os.path.join(event_folder, str(i).zfill(10) + ".npz") for i in
                                        range(event_index[1], event_index[2])]
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
                
                logger.update(f"{scene}_{difficulty}", psnr, ssim, lpips, flolpips, dists)
                logger.update(f"multi_{difficulty}", psnr, ssim, lpips, flolpips, dists)
                if isTestPer:
                    print(scene, "\'s ", psnr, "  ", ssim, " ", lpips, " ", flolpips, " ", dists)
                    
                save_name = os.path.join(val_folder, str(index).zfill(6) + ".png")
                pred_out = cv2.cvtColor(pred_out, cv2.COLOR_RGB2BGR)
                if isSave:
                    if isinstance(saveSpecificScene, list):
                        if scene in saveSpecificScene:
                            cv2.imwrite(save_name, pred_out)
                    else:
                        cv2.imwrite(save_name, pred_out)
        logger.print_summary(f"multi_{difficulty}")