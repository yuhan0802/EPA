import cv2
import h5py
import numpy as np
import os
import torch

from datasets.HQF import binary_search_h5_dset
from predict.pred_utils import get_voxel_and_mask, psnr_, ssim_
from util.event import EventSequence


def predict_hqf(model, nums_bins, test_scenes, device, save_path, multis=[1,3], isSave=False, isTestPer=False, 
                  saveSpecificScene=None):
    print("Start test HQF!")
    test_path = '/DATASSD1/HQF_H5/'
    for multi in multis:
        psnr_mul = []
        ssim_mul = []
        for scene in test_scenes:
            if scene != '':
                psnr_scene = []
                ssim_scene = []
                val_folder = os.path.join(save_path, scene + "_" + str(multi))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                if not os.path.exists(val_folder):
                    os.mkdir(val_folder)
                data_path = os.path.join(test_path, (scene+".h5"))
                h5_file = h5py.File(data_path, 'r')
                h, w = h5_file.attrs['sensor_resolution'][0:2]
                num_frames = h5_file.attrs["num_imgs"]
                index = 0
                frame_ts = []
                for img_name in h5_file['images']:
                    frame_ts.append(h5_file['images/{}'.format(img_name)].attrs['timestamp'])
                while (index + multi + 2) < num_frames:
                    img0 = h5_file['images']['image{:09d}'.format(index)][:]
                    img1 = h5_file['images']['image{:09d}'.format(index + multi + 1)][:]
                    img0 = torch.from_numpy(img0).float().unsqueeze(0).repeat(3,1,1)
                    img1 = torch.from_numpy(img1).float().unsqueeze(0).repeat(3,1,1)
                    for i in range(multi):
                        gt = h5_file['images']['image{:09d}'.format(index+i+1)][:]
                        gt = torch.from_numpy(gt).float().unsqueeze(0).repeat(3,1,1)
                        # gray_img0 = img0
                        # gray_img1 = img1
                        # gray_gt = gt
                        imgs = (torch.cat((img0, gt, img1), 0)/255.0).unsqueeze(0).to(device, non_blocking=True)
                        
                        # start = binary_search_h5_dset(h5_file['events/ts'], frame_ts[index])
                        # t = binary_search_h5_dset(h5_file['events/ts'], frame_ts[index+i+1])
                        # end = binary_search_h5_dset(h5_file['events/ts'], frame_ts[index+multi+1])
                        
                        # events_0t = load_events(h5_file, start, t)
                        # events_t1 = load_events(h5_file, t, end)
                        # try:
                        #     events_0t = EventSequence(events_0t, h, w)
                        # except:
                        #     events_0t = None
                        # try:
                        #     events_t1 = EventSequence(events_t1, h, w)
                        # except:
                        #     events_t1 = None
                        # event_voxel, mask = get_voxel_and_mask(nums_bins, events_0t, events_t1, h, w)
                        batch = {'img0': imgs[:, :3],
                                'gt': imgs[:, 3:6],
                                'img1': imgs[:, 6:9]}
                        
                        with torch.no_grad():
                            pred = model.inference(batch)
                        pred_out = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        gt = (gt.numpy()).transpose(1,2,0).astype(np.uint8)
                        psnr = psnr_(gt, pred_out)
                        ssim = ssim_(gt, pred_out)
                        psnr_scene.append(psnr)
                        psnr_mul.append(psnr)
                        ssim_mul.append(ssim)
                        
                        if isTestPer:
                            print(psnr, "  ", ssim)
                            
                        save_name = os.path.join(val_folder, "image"+str(index + i + 1).zfill(9)+".png")
                        if isSave:
                            if isinstance(saveSpecificScene, list):
                                if scene in saveSpecificScene:
                                    cv2.imwrite(save_name, pred_out)
                            else:
                                cv2.imwrite(save_name, pred_out)
                    index = index + multi
                print(scene, "\'s ", multi, " is ", np.array(psnr_scene).mean())
        print(multi, " is ", np.array(psnr_mul).mean())
        print(multi, " is ", np.array(ssim_mul).mean())
        
        
        
def load_events(file):
    tmp = np.load(file, allow_pickle=True)
    (x, y, timestamp, polarity) = (
        tmp["x"].astype(np.float64).reshape((-1,)),
        tmp["y"].astype(np.float64).reshape((-1,)),
        tmp["t"].astype(np.float64).reshape((-1,)),
        tmp["p"].astype(np.float32).reshape((-1,))
    )
    events = np.stack((x, y, timestamp, polarity), axis=-1)
    return events