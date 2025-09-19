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
    parser.add_argument('--epoch', default=40, type=int)
    parser.add_argument('--batch_size', default=32, type=int, help='minibatch size')
    parser.add_argument('--image_size', default=256, type=int, help='image size')
    parser.add_argument('--num_worker', default=24, type=int, help='num worker')
    parser.add_argument('--bins', default=8, type=int, help='number of time bins')
    parser.add_argument('--RESUME', default=False, type=bool, help='RESUME')
    parser.add_argument('--RESUME_EPOCH', default=5, type=int, help='RESUME_EPOCH')
    args = parser.parse_args()
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    model = Model(args)

    # Load the pre-trained generator's weights
    model.my_net.generator.load_state_dict(torch.load("xxx")['net'])

    train(model, args)