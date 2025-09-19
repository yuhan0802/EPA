import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import warnings

from predict.bsergb import predict_bsergb
from predict.aid import predict_aid
from predict.gopro import predict_gopro
from predict.hsergb import predict_hsergb
from predict.vimeo90k import predict_vimeo90k
from model.ourModel import Model

device = torch.device("cuda")
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='bsergb', type=str, help='dataset name')
    parser.add_argument('--bins', default=8, type=int, help='number of time bins')
    args = parser.parse_args()
    model = Model(args)

    # weights
    model.load_model("xxx")

    # save imgs
    val_root_path = "xxx"
    if args.dataset == 'gopro':
        save_path = val_root_path + args.dataset
        saveSpecificScene = None
        predict_gopro(model=model, bins=args.bins, device=device, save_path=save_path, multis=[15, 7]
                      , isSave=False, isTestPer=False, saveSpecificScene=saveSpecificScene)
    if args.dataset == 'hsergb':
        save_path = val_root_path + "hsergb"
        saveSpecificScene = None
        predict_hsergb(model, args.bins, device, save_path, multis=[5,7], isSave=False, isTestPer=False, 
                  saveSpecificScene=saveSpecificScene)
    if args.dataset == 'vimeo90k':
        save_path = val_root_path + "90k"
        saveIndexRange = [0, 2000]
        predict_vimeo90k(model, args.bins, device, save_path, isSave=False, isTestPer=True, 
                  saveIndexRange=saveIndexRange)
    if args.dataset == 'bsergb':
        save_path = val_root_path + args.dataset
        saveSpecificScene = None
        predict_bsergb(model, args.bins, device, save_path, multis=[1], isSave=False, isTestPer=False, 
                  saveSpecificScene=saveSpecificScene)
    if args.dataset == 'aid':
        save_path = val_root_path + args.dataset
        saveSpecificScene = None
        predict_aid(model, args.bins, device, save_path, multis=[7], isSave=True, isTestPer=False, 
                  saveSpecificScene=saveSpecificScene)
