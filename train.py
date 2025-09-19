import math
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings


warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_path = 'train'


def get_learning_rate(step, args):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6


def train(model, args):
    writer = SummaryWriter('train')
    start_epoch = 0
    step = 0
    nr_eval = 0


    # Load training data
    train_dataset = None

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                  pin_memory=True,
                                  drop_last=True, shuffle=True)
    args.step_per_epoch = train_dataloader.__len__()

    print('training...')
    time_stamp = time.time()
    if args.RESUME:
        step = args.step_per_epoch * args.RESUME_EPOCH
        nr_eval = args.RESUME_EPOCH
        start_epoch = args.RESUME_EPOCH
    for epoch in range(start_epoch, args.epoch):
        for i, data in enumerate(train_dataloader):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data = [tensor.to(device, non_blocking=True) for tensor in data]
            pure_gt, img0, gt, img1, event_0t_voxel, event_t1_voxel, event_1t_voxel = data
            learning_rate = get_learning_rate(step, args) * 1 / 8

            batch = {'img0': img0,
                     'gt': gt,
                     'pure_gt': pure_gt,
                     'img1': img1,
                     'e0t': event_0t_voxel,
                     'e1t': event_1t_voxel,
                     'et1': event_t1_voxel}

            pred, loss = model.update(batch, learning_rate)
            loss = loss.item()
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 500 == 1:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/l1', loss, step)
            if step % 1000 == 1:
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pure_gt = (pure_gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                for i in range(4):
                    imgs = np.concatenate((pred[i], pure_gt[i]), 1)
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                writer.flush()
            print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e}'.format(epoch, i, args.step_per_epoch,
                                                                            data_time_interval, train_time_interval,
                                                                            loss))
            step += 1
        nr_eval += 1
        if nr_eval % 5 == 0:
            model.save_checkpoint('not', nr_eval)
        if nr_eval % 10 == 0:
            model.save_checkpoint('not', nr_eval)
