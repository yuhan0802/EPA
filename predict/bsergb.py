import cv2
import numpy as np
import os
import torch
from numba import jit

from predict.pred_utils import get_imgs, looking_for_event_index_by_timestamp, psnr_, ssim_


@jit(nopython=True)
def trilinear_alloc_values(voxel, d_x, d_y, d_t, d_p, h, w, tstep, tstart):
    d_x_low, d_y_low = int(d_x), int(d_y)
    d_t_cur = (d_t-tstart)*tstep
    d_t_low = int(d_t_cur)

    x_weight = d_x - d_x_low
    y_weight = d_y - d_y_low
    t_weight = d_t_cur - d_t_low
    pv = d_p * 2 - 1
    if d_y_low < h and d_x_low < w:
        voxel[d_t_low, d_y_low, d_x_low] += (1 - x_weight) * (1 - y_weight) * pv * (1 - t_weight)
        voxel[d_t_low+1, d_y_low, d_x_low] += (1 - x_weight) * (1 - y_weight) * pv * t_weight
    if d_y_low + 1 < h and d_x_low < w:
        voxel[d_t_low, d_y_low + 1, d_x_low] += (1 - x_weight) * y_weight * pv * (1 - t_weight)
        voxel[d_t_low+1, d_y_low + 1, d_x_low] += (1 - x_weight) * y_weight * pv * t_weight
    if d_x_low + 1 < w and d_y_low < h:
        voxel[d_t_low, d_y_low, d_x_low + 1] += (1 - y_weight) * x_weight * pv * (1 - t_weight)
        voxel[d_t_low+1, d_y_low, d_x_low + 1] += (1 - y_weight) * x_weight * pv * t_weight
    if d_y_low + 1 < h and d_x_low + 1 < w:
        voxel[d_t_low, d_y_low + 1, d_x_low + 1] += x_weight * y_weight * pv * (1 - t_weight)
        voxel[d_t_low+1, d_y_low + 1, d_x_low + 1] += x_weight * y_weight * pv * t_weight
    return


@jit(nopython=True)
def sample_events_to_grid(voxel_channels, h, w, x, y, t, p, hs, ws, tleft):
    events = []
    x = (x-ws) / (19968 * w / h) * (w - 1)
    y = (y-hs) / 19968 * (h - 1)
    ori_left_voxel = np.zeros((voxel_channels, h, w), dtype=np.float32)
    right_voxel = np.zeros((voxel_channels, h, w), dtype=np.float32)
    # if len(t) == 0:
    #     return voxel
    t_start = t[0]
    t_end = t[-1]
    # t_step = (t_end - t_start + 1) / voxel_channels
    # Compute left step
    tstep_left = float(voxel_channels-1)/float(tleft-t_start+1)
    tstep_right = float(voxel_channels-1) / float(t_end-tleft+1)
    for d in range(len(x)):
        d_x, d_y, d_t, d_p = x[d], y[d], t[d], p[d]
        if d_t < tleft:
            trilinear_alloc_values(ori_left_voxel, d_x, d_y, d_t, d_p, h, w, tstep_left, t_start)
        else:
            trilinear_alloc_values(right_voxel, d_x, d_y, d_t, d_p, h, w, tstep_right, tleft)
    # 对 t_end - 1 的事件进行反转
    reversed_right_voxel = -right_voxel[::-1]

    return ori_left_voxel, right_voxel, reversed_right_voxel


indexing_skip_ind = {
    'basket_09':[31, 32, 33, 34],
    'may29_rooftop_handheld_02':[17, 70],
    'may29_rooftop_handheld_03':[306],
    'may29_rooftop_handheld_05':[21],
}


def predict_bsergb(model, num_bins, device, save_path, multis=[1,3,5], isSave=False, isTestPer=False, 
                  saveSpecificScene=None):
    print("Start test BSERGB!")
    test_path = '/DATASSD1/BSERGB/1_TEST'
    for multi in multis:
        interp_ratio = multi + 1
        psnr_mul = []
        ssim_mul = []
        for scene in os.listdir(test_path):
            if scene in ["acquarium_08", "ball_05", "basket_08", "candies_03"]:
                psnr_all = []
                ssim_all = []
                val_folder = os.path.join(save_path, scene + '_' + str(multi))
                img_folder = os.path.join(test_path, scene, 'images')
                img_names = os.listdir(img_folder)
                img_names.sort()
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                if not os.path.exists(val_folder):
                    os.makedirs(val_folder, exist_ok=True)
                img_paths = [os.path.join(img_folder, i) for i in img_names if i.endswith("png") or i.endswith("jpg")]
                img_paths.sort()
                h, w, _ = cv2.imread(img_paths[0]).shape
                index = 0
                while (index + multi + 3) < len(img_names):
                    imgs = get_imgs(img_paths, multi, index)
                    for i in range(multi):
                        gt = cv2.imread(img_paths[index + i + 1])
                        img_index0, gt_index, img_index1 = index, index + i + 1, index + multi + 1
                        
                        gt_tensor = torch.from_numpy(gt.copy()).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True) / 255.
                        skip_flag = False
                        if scene in indexing_skip_ind:
                            for skip_index in indexing_skip_ind[scene]:
                                if img_index0 <= skip_index < img_index1:
                                    skip_flag = True
                                    break
                        if skip_flag:
                            continue
                        
                        # event_voxel, mask = looking_for_event_index_by_timestamp(
                        #     os.path.join(test_path, scene), num_bins,
                        #     img_index0, gt_index,
                        #     img_index1, h, w, 240, real=True, bsergb=True)
                        
                        # event_voxel = event_voxel.to(device, non_blocking=True)
                        # mask = mask.to(device, non_blocking=True)
                        batch = {'img0': imgs[:, :3],
                                'gt': gt_tensor,
                                'img1': imgs[:, 3:6]}
                        with torch.no_grad():
                            pred = model.inference(batch)
                        pred = torch.clamp(pred, 0, 1)
                        pred_out = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        psnr = psnr_(gt, pred_out)
                        ssim = ssim_(gt, pred_out)
                        psnr_all.append(psnr)
                        psnr_mul.append(psnr)
                        ssim_all.append(ssim)
                        ssim_mul.append(ssim)
                        if isTestPer:
                            print(scene, "\'s ", gt_index, "\'s PSNR is ", np.array(psnr_all).mean(), " SSIM is ", np.array(ssim_all).mean())
                        cv2.imwrite("test.png", pred_out)
                        save_name = os.path.join(val_folder, img_names[index + i + 1])
                        if isSave:
                            if isinstance(saveSpecificScene, list):
                                if scene in saveSpecificScene:
                                    cv2.imwrite(save_name, pred_out)
                            else:
                                cv2.imwrite(save_name, pred_out)
                    index = index + multi
                print(scene, "\'s ", multi, "\'s PSNR is ", np.array(psnr_all).mean())
                print(scene, "\'s ", multi, "\'s SSIM is ", np.array(ssim_all).mean())
        print(multi, "\'s PSNR is ", np.array(psnr_mul).mean())
        print(multi, "\'s SSIM is ", np.array(ssim_mul).mean())
