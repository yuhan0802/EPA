import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import warnings

from predict.adobe import predict_adobe
from predict.bsergb import predict_bsergb
from predict.middlebury import predict_middlebury
from predict.gopro import predict_gopro
from predict.hqf import predict_hqf
from predict.hsergb import predict_hsergb
from predict.hsergb_all import predict_hsergb_all
from predict.sunfilm import predict_snufilm
from predict.vimeo90k import predict_vimeo90k
from model.ourModel import Model

device = torch.device("cuda")
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='bsergb', type=str, help='dataset name')
    parser.add_argument('--RESUME', default=False, type=bool, help='RESUME')
    parser.add_argument('--RESUME_EPOCH', default=40, type=int, help='RESUME_EPOCH')
    parser.add_argument('--bins', default=8, type=int, help='number of time bins')
    parser.add_argument('--sample_factor', default=1, type=int, help='sampling proportion, 1/factor')
    parser.add_argument('--mask_patch_size', default=32, type=int, help='different mask patch size')
    args = parser.parse_args()
    model = Model(args)
    
    model.load_model("/home/lyh/PycharmProjects_lyh/work3/v6/train/checkpoint/train_ckpt_10.pth")
    
    val_root_path = "/home/lyh/PycharmProjects_lyh/imgs_pred/ours_work3/"
    if args.dataset == 'gopro':
        save_path = val_root_path + args.dataset
        saveSpecificScene = None
        predict_gopro(model=model, bins=args.bins, device=device, save_path=save_path, multis=[7]
                      , isSave=True, isTestPer=False, saveSpecificScene=saveSpecificScene)
    if args.dataset == 'adobe':
        save_path = val_root_path + args.dataset
        saveSpecificScene = None
        predict_adobe(model, args.bins, device, save_path, multis=[7,15], isSave=True, isTestPer=True, 
                  saveSpecificScene=saveSpecificScene)
    if args.dataset == 'middlebury':
        save_path = val_root_path + args.dataset
        saveIndexRange = None
        predict_middlebury(model, args.bins, device, save_path, multis=[1,3], isSave=False, isTestPer=False, 
                  saveIndexRange=saveIndexRange)
    if args.dataset == 'hqf':
        save_path = val_root_path + args.dataset
        test_scenes = ["poster_pillar_1", "slow_and_fast_desk", "bike_bay_hdr", "desk"]
        saveSpecificScene = None
        predict_hqf(model, args.bins, test_scenes, device, save_path, multis=[1,3], isSave=False, isTestPer=True, 
                  saveSpecificScene=saveSpecificScene)
    if args.dataset == 'hsergb_all':
        save_path = val_root_path + args.dataset
        saveSpecificScene = None
        predict_hsergb_all(model, args.bins, device, save_path, multis=[5,7], isSave=False, isTestPer=False, 
                  saveSpecificScene=saveSpecificScene)
    if args.dataset == 'hsergb_finetune':
        save_path = val_root_path + "hsergb"
        saveSpecificScene = None
        predict_hsergb(model, args.bins, device, save_path, multis=[5,7], isSave=False, isTestPer=True, 
                  saveSpecificScene=saveSpecificScene)
    if args.dataset == 'snufilm':
        save_path = val_root_path + args.dataset
        saveSpecificScene = None
        predict_snufilm(model, args.bins, device, save_path, difficulties=['hard', 'extreme'], isSave=True, isTestPer=False, 
                  saveSpecificScene=saveSpecificScene)
    if args.dataset == 'vimeo90k':
        save_path = val_root_path + "90k"
        saveIndexRange = [0, 2000]
        predict_vimeo90k(model, args.bins, device, save_path, isSave=False, isTestPer=False, 
                  saveIndexRange=saveIndexRange)
    if args.dataset == 'bsergb':
        save_path = val_root_path + args.dataset
        saveSpecificScene = None
        predict_bsergb(model, args.bins, device, save_path, multis=[1,3], isSave=False, isTestPer=False, 
                  saveSpecificScene=saveSpecificScene)
