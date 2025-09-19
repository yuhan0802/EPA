import cv2
import numpy as np
import os
import torch
from PIL import Image
from predict.pred_utils import GroupedMetricLogger, dists_, flolpips_, get_imgs, get_voxel_and_mask, looking_for_event_index_by_timestamp, lpips_, psnr_, ssim_
import torchvision.transforms as transforms
from util.event import EventSequence

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
        
to_tensor = transforms.Compose([
    transforms.ToTensor()
])

def predict_hsergb(model, num_bins, device, save_path, multis=[5,7], isSave=False, isTestPer=False, 
                  saveSpecificScene=None):
    print("Start test HS-ERGB!")
    close_scenes = ["candle", "fountain_bellevue2", "water_bomb_eth_01"]
    close_scenes_align = ["spinning_umbrella", "baloon_popping", "confetti", "fountain_schaffhauserplatz_02",
                            "spinning_plate", "water_bomb_floor_01"]

    far_scenes = ["kornhausbruecke_letten_random_04", "sihl_03"]
    far_scenes_align = ["bridge_lake_01", "bridge_lake_03", "lake_01", "lake_03"]
    logger = GroupedMetricLogger()
    for multi in multis:
        for distance in ['close', 'far']:
            test_path = os.path.join('/data/hsergb', distance, 'test')
            val_path = save_path
            for align in ['not', 'finish']:
                if distance == 'close' and align == 'finish':
                    scenes = close_scenes
                elif distance == 'close' and align == 'not':
                    scenes = close_scenes_align
                elif distance == 'far' and align == 'finish':
                    scenes = far_scenes
                else:
                    scenes = far_scenes_align
                for scene in scenes:
                    if scene != "":
                        val_folder = os.path.join(val_path, distance, scene + "_" + str(multi))
                        img_folder = os.path.join(test_path, scene, 'images_corrected')
                        event_folder = os.path.join(test_path, scene, 'events_aligned')
                        img_names = os.listdir(img_folder)
                        img_names.sort()
                        os.makedirs(val_folder, exist_ok=True)
                        new_img_names = []
                        for name in img_names:
                            if name.endswith("png"):
                                new_img_names.append(name)
                        new_img_names.sort()
                        img_paths = [os.path.join(img_folder, i) for i in new_img_names]
                        event_names = os.listdir(event_folder)
                        event_names.sort()
                        h, w, _ = cv2.imread(img_paths[0]).shape
                        index = 0
                        while (index + multi + 3) < len(new_img_names):
                            img0, img1 = get_imgs(img_paths, multi, index)
                            for i in range(multi):
                                gt = Image.open(img_paths[index + i + 1])
                                w, h = gt.size
                                pure_gt = gt.copy()
                                pure_gt = to_tensor(pure_gt).unsqueeze(0).to(device, non_blocking=True)
                                gt = normalize(gt).unsqueeze(0).to(device, non_blocking=True)
                                
                                img_index0, gt_index, img_index1 = index, index + i + 1, index + multi + 1
                                z = 1 if align != 'finish' else 0
                                start_ind = img_index0 + z
                                gt_ind = gt_index + z
                                end_ind = img_index1 + z
                                
                                events_before_names = [os.path.join(event_folder, event_names[i]) for i in range(start_ind, gt_ind)]
                                events_after_names = [os.path.join(event_folder, event_names[i]) for i in range(gt_ind, end_ind)]
                                try:
                                    events_0t = EventSequence.from_npz_files(events_before_names, h, w, hsergb=True, size=[0, 0, h, w])
                                except:
                                    events_0t = None
                                try:
                                    events_t1 = EventSequence.from_npz_files(events_after_names, h, w, hsergb=True, size=[0, 0, h, w])
                                except:
                                    events_t1 = None
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
                                lpipss = lpips_(pure_gt, pred_out)
                                logger.update(f"{distance}_{scene}_{multi}", psnr, ssim, lpipss, flolpips, dists)
                                logger.update(f"{distance}_{multi}", psnr, ssim, lpipss, flolpips, dists)
                                logger.update(f"{multi}", psnr, ssim, lpipss, flolpips, dists)
                                if isTestPer:
                                    print(psnr, " ", ssim, " ", lpipss, " ", flolpips, " ", dists)
                                    
                                save_name = os.path.join(val_folder, img_names[index + i + 1])
                                if isSave:
                                    save_img = Image.fromarray(pred_out)
                                    if isinstance(saveSpecificScene, list):
                                        if scene in saveSpecificScene:
                                            save_img.save(save_name)
                                    else:
                                        save_img.save(save_name)
                            index = index + multi
                        logger.print_summary(f"{distance}_{scene}_{multi}")
            logger.print_summary(f"{distance}_{multi}")
        logger.print_summary(f"{multi}")