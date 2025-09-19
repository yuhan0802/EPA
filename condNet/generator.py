import accelerate
import torch
from math import log

from torch import nn

from condNet import thops
from condNet.normalizing_flow import CondFlowNet
from model.submodules import conv5x5_resblock_one


class Network(torch.nn.Module):
    def __init__(self, cond_c):
        super().__init__()
        self.condFLownet = CondFlowNet(cond_c, with_bn=False, train_1x1=True, K=4)

    def normalize(self, x, reverse=False):
        # x in [0, 1]
        if not reverse:
            return x * 2 - 1
        else:
            return (x + 1) / 2

    def forward(self, gt=None, zs=None, inps=[], code="encode"):
        if code == "encode":
            return self.encode(gt, inps)
        elif code == "decode":
            return self.decode(zs, inps)
        else:
            return self.encode_decode(gt, inps, zs=zs)

    def encode(self, gt, inps: list, time: float = 0.5):
        img0, img1 = [self.normalize(x) for x in inps[:2]]
        gt = self.normalize(gt)
        cond = [img0, img1] + inps[-2:]
        pixels = thops.pixels(gt)
        conds, smasks = self.get_cond(cond, time=time)

        # add random noise before normalizing flow net
        loss = 0.0
        if self.training:
            gt = gt + ((torch.rand_like(gt, device=gt.device) - 0.5) / 255.0)
            loss += -log(255.0) * pixels
        log_p, log_det, zs = self.condFLownet(gt, conds)

        loss /= float(log(2) * pixels)
        log_p /= float(log(2) * pixels)
        log_det /= float(log(2) * pixels)
        nll = -(loss + log_det + log_p)
        return nll, zs, smasks

    def decode(self, z_list: list, inps: list, time: float = 0.5):
        img0, img1 = [self.normalize(x) for x in inps[:2]]
        cond = [img0, img1] + inps[-2:]

        conds, smasks = self.get_cond(cond, time=time)
        pred = self.condFLownet(z_list, conds, reverse=True)
        pred = self.normalize(pred, reverse=True)
        return pred, smasks

    def encode_decode(self, gt, inps: list, time: float = 0.5, zs=None):
        img0, img1 = [self.normalize(x) for x in inps[:2]]
        gt = self.normalize(gt)
        cond = [img0, img1] + inps[-2:]
        pixels = thops.pixels(gt)
        conds, smasks = self.get_cond(cond, time=time)

        # encode first
        loss = 0.0
        if self.training:
            gt = gt + ((torch.rand_like(gt, device=gt.device) - 0.5) / 255.0)
            loss += -log(255.0) * pixels
        log_p, log_det, zs_gt = self.condFLownet(gt, conds)
        loss /= float(log(2) * pixels)
        log_p /= float(log(2) * pixels)
        log_det /= float(log(2) * pixels)
        nll = -(loss + log_det + log_p)

        # decode next
        if zs is None:
            heat = torch.sqrt(torch.var(torch.cat([x.flatten() for x in zs_gt])))
            zs = self.get_z(heat, img0.shape[-2:], img0.shape[0], img0.device)
        pred = self.condFLownet(zs, conds, reverse=True)
        pred = self.normalize(pred, reverse=True)
        return nll, pred, smasks

    def get_z(self, heat: float, img_size: tuple, batch: int, device: str):
        def calc_z_shapes(img_size, n_levels):
            h, w = img_size
            z_shapes = []
            channel = 3

            for _ in range(n_levels - 1):
                h //= 2
                w //= 2
                channel *= 2
                z_shapes.append((channel, h, w))
            h //= 2
            w //= 2
            z_shapes.append((channel * 4, h, w))
            return z_shapes

        z_list = []
        z_shapes = calc_z_shapes(img_size, 3)
        for z in z_shapes:
            z_new = torch.randn(batch, *z, device=device) * heat
            z_list.append(z_new)
        return z_list