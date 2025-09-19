import torch
from torch import nn

from math import log
from condNet import thops
from condNet.msfusion import MultiscaleFuse
from condNet.normalizing_flow import CondFlowNet
from model.blocks import EncoderImage
from util.utils_func import DINOResNetFeatureExtractor


class netGenerator(nn.Module):
    def __init__(self, cond_c, training):
        super(netGenerator, self).__init__()
        self.dino_fea_extractor = DINOResNetFeatureExtractor(torch.hub.load('facebookresearch/dino:main', 'dino_resnet50'))
        self.frame_encoder = EncoderImage(cond_c)
        self.generator = CondFlowNet(cond_c, with_bn=False, train_1x1=True, K=16)
        self.training = training

    def get_cond(self, img):
        # low, middle, high
        dino_feat = self.dino_fea_extractor(img)
        conds = self.frame_encoder(img, dino_feat)
        return conds

    def normalize(self, x, reverse=False):
        # x in [0, 1]
        if not reverse:
            return x * 2 - 1
        else:
            return (x + 1) / 2

    def forward(self, gt, img0, img1, zs=None, code="encode", conds=None):
        if code == "encode":
            return self.encode(gt, img0, img1)
        elif code == "decode":
            return self.decode(zs, conds)
        else:
            return self.encode_decode(gt, img0, img1, zs=zs)

    def encode(self, gt, img0, img1):
        img0 = self.normalize(img0)
        img1 = self.normalize(img1)
        gt = self.normalize(gt)
        cond = [img0, img1, gt]
        pixels = thops.pixels(gt)
        conds = self.get_cond(cond)

        # add random noise before normalizing flow net
        loss = 0.0
        if self.training:
            gt = gt + ((torch.rand_like(gt, device=gt.device) - 0.5) / 255.0)
            loss += -log(255.0) * pixels
        log_p, log_det, zs = self.generator(gt, conds)

        loss /= float(log(2) * pixels)
        log_p /= float(log(2) * pixels)
        log_det /= float(log(2) * pixels)
        nll = -(loss + log_det + log_p)
        return nll, zs

    def decode(self, z_list, conds):
        pred = self.generator(z_list, conds, reverse=True)
        # pred = self.normalize(pred, reverse=True)
        return pred

    def encode_decode(self, gt, img0, img1, zs=None):
        img0 = self.normalize(img0)
        img1 = self.normalize(img1)
        gt = self.normalize(gt)

        pixels = thops.pixels(gt)
        conds = self.get_cond(gt)

        # encode first
        loss = 0.0
        if self.training:
            gt = gt + ((torch.rand_like(gt, device=gt.device) - 0.5) / 255.0)
            loss += -log(255.0) * pixels
        log_p, log_det, zs_gt = self.generator(gt, conds)
        loss /= float(log(2) * pixels)
        log_p /= float(log(2) * pixels)
        log_det /= float(log(2) * pixels)
        nll = -(loss + log_det + log_p)

        # decode next
        if zs is None:
            heat = torch.sqrt(torch.var(torch.cat([x.flatten() for x in zs_gt])))
            zs = self.get_z(heat, img0.shape[-2:], img0.shape[0], img0.device)
        pred = self.generator(zs, conds, reverse=True)
        pred = self.normalize(pred, reverse=True)
        return nll, pred

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


if __name__ == '__main__':
    device = torch.device("cpu")
    imgs = torch.randn([2, 3, 224, 224]).to(device, non_blocking=True)