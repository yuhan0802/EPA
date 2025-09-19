import cv2
import numpy as np
import os
import torch

from predict.pred_utils import get_imgs, looking_for_event_index_by_timestamp, psnr_, ssim_



def predict_middlebury(model, num_bins, device, save_path, multis=[1,3], isSave=False, isTestPer=False, 
                  saveIndexRange=None):
    print("Start test Middlebury!")
    test_path = '/data/Middleburry_all/'
    for multi in multis:
        psnr_mul = []
        ssim_mul = []
        for scene in os.listdir(test_path):
            psnr_scene = []
            ssim_scene = []
            val_folder = os.path.join(save_path, scene + "_" + str(multi))
            img_folder = os.path.join(test_path, scene, 'imgs')
            img_names = os.listdir(img_folder)
            img_names.sort()
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if not os.path.exists(val_folder):
                os.mkdir(val_folder)
            img_paths = [os.path.join(img_folder, i) for i in img_names]
            h, w, _ = cv2.imread(img_paths[0]).shape
            index = 0
            timestamps = [((i + 1) / (multi + 1)) for i in range(multi)]
            while (index + multi + 1) < len(img_names):
                imgs = get_imgs(img_paths, multi, index)
                for i in range(multi):
                    gt = cv2.imread(img_paths[index + i + 1])
                    img_index0, gt_index, img_index1 = index, index + i + 1, index + multi + 1
                    event_voxel, mask = looking_for_event_index_by_timestamp(os.path.join(test_path, scene), num_bins,
                                                                                        img_index0, gt_index,
                                                                                        img_index1, h, w, 25)
                    event_voxel = event_voxel.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    with torch.no_grad():
                        pred = model.inference(imgs, event_voxel, mask)
                    pred_out = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    psnr = psnr_(gt, pred_out)
                    ssim = ssim_(gt, pred_out)
                    psnr_scene.append(psnr)
                    ssim_scene.append(ssim)
                    psnr_mul.append(psnr)
                    ssim_mul.append(ssim)
                    if isTestPer:
                        print(psnr, "  ", ssim)
                        
                    save_name = os.path.join(val_folder, img_names[index + i + 1])
                    if isSave:
                        if isinstance(saveIndexRange, list):
                            if index >= saveIndexRange[0] and index <= saveIndexRange[1]:
                                cv2.imwrite(save_name, pred_out)
                        else:
                            cv2.imwrite(save_name, pred_out)
                index = index + multi
            print(scene, "\'s ", multi, " psnr is ", np.array(psnr_scene).mean(), ", ssim is ", np.array(ssim_scene).mean())
        print(multi, " is ", np.array(psnr_mul).mean())
        print(multi, " is ", np.array(ssim_mul).mean())