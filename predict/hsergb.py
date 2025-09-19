import cv2
import numpy as np
import os
import torch

from predict.pred_utils import get_imgs, get_voxel_and_mask, psnr_, ssim_
from util.event import EventSequence

def predict_hsergb(model, num_bins, device, save_path, multis=[5,7], isSave=False, isTestPer=False, 
                  saveSpecificScene=None):
    print("Start test HS-ERGB!")
    close_scenes = ["candle", "fountain_bellevue2", "water_bomb_eth_01"]
    close_scenes_align = ["spinning_umbrella", "baloon_popping", "confetti", "fountain_schaffhauserplatz_02",
                            "spinning_plate", "water_bomb_floor_01"]

    far_scenes = ["kornhausbruecke_letten_random_04", "sihl_03"]
    far_scenes_align = ["bridge_lake_01", "bridge_lake_03", "lake_01", "lake_03"]
    for multi in multis:
        psnr_mul = []
        ssim_mul = []
        timestamps = [((i + 1) / (multi + 1)) for i in range(multi)]
        for distance in ['close', 'far']:
            psnr_dis = []
            ssim_dis = []
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
                        psnr_scene = []
                        ssim_scene = []
                        val_folder = os.path.join(val_path, distance, scene + "_" + str(multi))
                        img_folder = os.path.join(test_path, scene, 'images_corrected')
                        event_folder = os.path.join(test_path, scene, 'events_aligned')
                        img_names = os.listdir(img_folder)
                        img_names.sort()
                        if not os.path.exists(val_folder):
                            os.mkdir(val_folder)
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
                            imgs = get_imgs(img_paths, multi, index)
                            for i in range(multi):
                                gt = cv2.imread(img_paths[index + i + 1])
                                # img_index0, gt_index, img_index1 = index, index + i + 1, index + multi + 1
                                # z = 1 if align != 'finish' else 0
                                # start_ind = img_index0 + z
                                # gt_ind = gt_index + z
                                # end_ind = img_index1 + z
                                
                                # events_before_names = [os.path.join(event_folder, event_names[i]) for i in range(start_ind, gt_ind)]
                                # events_after_names = [os.path.join(event_folder, event_names[i]) for i in range(gt_ind, end_ind)]
                                # try:
                                #     events_0t = EventSequence.from_npz_files(events_before_names, h, w, hsergb=True, size=[0, 0, h, w])
                                # except:
                                #     events_0t = None
                                # try:
                                #     events_t1 = EventSequence.from_npz_files(events_after_names, h, w, hsergb=True, size=[0, 0, h, w])
                                # except:
                                #     events_t1 = None
                                # event_voxel, mask = get_voxel_and_mask(num_bins, events_0t, events_t1, h, w)
                                batch = {'img0': imgs[:, :3],
                                        'gt': torch.from_numpy(gt.copy()).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True) / 255.,
                                        'img1': imgs[:, 3:6]}
                                with torch.no_grad():
                                    pred = model.inference(batch)

                                pred_out = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                                psnr = psnr_(gt, pred_out)
                                ssim = ssim_(gt, pred_out)
                                psnr_scene.append(psnr)
                                ssim_scene.append(ssim)
                                psnr_dis.append(psnr)
                                ssim_dis.append(ssim)
                                psnr_mul.append(psnr)
                                ssim_mul.append(ssim)
                                if isTestPer:
                                    print(psnr, "  ", ssim)
                                    
                                save_name = os.path.join(val_folder, img_names[index + i + 1])
                                if isSave:
                                    if isinstance(saveSpecificScene, list):
                                        if scene in saveSpecificScene:
                                            cv2.imwrite(save_name, pred_out)
                                    else:
                                        cv2.imwrite(save_name, pred_out)
                            index = index + multi
                        print(multi, "_", distance, "_", scene, " \'s PSNR: ", np.array(psnr_scene).mean(),
                                " SSIM: ", np.array(ssim_scene).mean())
            print(multi, "_", distance, " \'s PSNR: ", np.array(psnr_dis).mean(),
                    " SSIM: ", np.array(ssim_dis).mean())
        print(multi, " \'s PSNR: ", np.array(psnr_mul).mean(),
                " SSIM: ", np.array(ssim_mul).mean())