import cv2
import numpy as np
import os
import torch
from PIL import Image
from predict.pred_utils import GroupedMetricLogger, flolpips_, get_imgs, looking_for_event_index_by_timestamp, lpips_, psnr_, ssim_
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

def predict_gopro(model, bins, device, save_path, multis=[7,15], isSave=False, isTestPer=False, 
                  saveSpecificScene=None):
    print("Start test GOPRO!")
    h, w = 720, 1280
    test_path = '/DATASSD1/GOPRO/test'
    val_path = save_path
    logger = GroupedMetricLogger()
    
    for multi in multis:    
        for scene in os.listdir(test_path):
            if scene != '':
                val_folder = os.path.join(val_path, scene + '_' + str(multi))
                img_folder = os.path.join(test_path, scene, 'imgs')
                img_names = os.listdir(img_folder)
                img_names.sort()

                os.makedirs(val_folder, exist_ok=True)
                
                img_paths = [os.path.join(img_folder, i) for i in img_names]
                index = 0
                
                while (index + multi + 1) < len(img_names):
                    img0, img1 = get_imgs(img_paths, multi, index)
                    for i in range(multi):
                        gt = Image.open(img_paths[index + i + 1])
                        w, h = gt.size
                        pure_gt = gt.copy()
                        pure_gt = to_tensor(pure_gt).unsqueeze(0).to(device, non_blocking=True)
                        gt = normalize(gt).unsqueeze(0).to(device, non_blocking=True)
                        
                        img_index0, gt_index, img_index1 = index, index + i + 1, index + multi + 1
                        event_voxel, mask = looking_for_event_index_by_timestamp(os.path.join(test_path, scene), bins,
                                                                                    img_index0, gt_index,
                                                                                    img_index1, h, w, 240)
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
                        pred_out = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        pure_gt = (pure_gt[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        psnr = psnr_(pure_gt, pred_out)
                        ssim = ssim_(pure_gt, pred_out)
                        lpips = lpips_(pure_gt, pred_out)
                        
                        logger.update(f"{scene}_{multi}", psnr, ssim, lpips, flolpips)
                        logger.update(f"multi_{multi}", psnr, ssim, lpips, flolpips)
                        
                        if isTestPer:
                            print(psnr, "  ", ssim, " ", lpips, " ", flolpips)
                            
                        save_name = os.path.join(val_folder, img_names[index + i + 1])
                        if isSave:
                            if isinstance(saveSpecificScene, list):
                                if scene in saveSpecificScene:
                                    cv2.imwrite(save_name, pred_out)
                            else:
                                cv2.imwrite(save_name, pred_out)
                    index = index + multi
                logger.print_summary(f"{scene}_{multi}")
        logger.print_summary(f"multi_{multi}")