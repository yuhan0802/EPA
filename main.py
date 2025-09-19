import argparse
import random
import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model.ourNet import ourNet
from train import train
from model.ourModel import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int, help='minibatch size')
    parser.add_argument('--image_size', default=256, type=int, help='image size')
    parser.add_argument('--test_batch_size', default=8, type=int, help='test minibatch size')
    parser.add_argument('--num_worker', default=24, type=int, help='num worker')
    parser.add_argument('--bins', default=8, type=int, help='number of time bins')
    parser.add_argument('--RESUME', default=False, type=bool, help='RESUME')
    parser.add_argument('--RESUME_EPOCH', default=5, type=int, help='RESUME_EPOCH')
    parser.add_argument('--train_dateset', default='vimeo', type=str, help='vimeo/hqf/hsergb')
    parser.add_argument('--sample_factor', default=2, type=int, help='sampling proportion, 1/factor')
    parser.add_argument('--mask_patch_size', default=32, type=int, help='different mask patch size')
    args = parser.parse_args()
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    model = Model(args)
    model.load_model("/home/lyh/PycharmProjects_lyh/work3/v6/train/checkpoint/90k_370099.pth")
    model.my_net.generator.load_state_dict(torch.load("/home/lyh/PycharmProjects_lyh/work3/generator_versions/gv2/train/checkpoint/train_ckpt_20.pth", weights_only=False)['net'])
    train(model, args)


# if __name__ == '__main__':
#     device = torch.device("cpu")
#     img = torch.randn([2, 3, 256, 256]).to(device, non_blocking=True)
#     e = torch.randn([2, 8, 256, 256]).to(device, non_blocking=True)
#     batch = {
#         'img0':img,
#         'gt':img,
#         'img1':img,
#         'e0t':e,
#         'e1t':e,
#     }
    
#     net = ourNet(8, [32, 64, 96])
    
#     list = net(batch)