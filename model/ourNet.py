import torch
from torch import nn

from model.netGenerator import netGenerator
from model.blocks import FusionNet, alignNet


class ourNet(nn.Module):
    def __init__(self, bins, cond_c):
        super(ourNet, self).__init__()
        self.generator = netGenerator(cond_c, training=True)
        self.generator.eval()
        for p in self.generator.parameters():
            p.requires_grad = False
            
        self.fea_align = alignNet(bins, cond_c)
        self.fusion = FusionNet(cond_c)
        
    def forward(self, batch, heat=0.3):
        F_0s = self.generator.get_cond(batch['img0'])
        Gts = self.generator.get_cond(batch['gt'])
        F_1s = self.generator.get_cond(batch['img1'])

        # Bidirectional Event-Guided Alignment
        Gts_pred_0 = self.fea_align(F_0s, F_1s, batch['e0t'])
        Gts_pred_1 = self.fea_align(F_1s, F_0s, batch['e1t'])
        Gts_pred = self.fusion(Gts_pred_0, Gts_pred_1)

        # Generator
        zs = self.generator.get_z(heat, batch['img0'].shape[-2:], batch['img0'].shape[0], batch['img0'].device)
        pred = self.generator.decode(zs, Gts_pred)

        return Gts, Gts_pred, pred, Gts_pred_0, Gts_pred_1


if __name__ == '__main__':
    device = torch.device("cpu")
    img = torch.randn([2, 3, 256, 256]).to(device, non_blocking=True)
    e = torch.randn([2, 8, 256, 256]).to(device, non_blocking=True)
    batch = {
        'img0':img,
        'gt':img,
        'img1':img,
        'e0t':e,
        'e1t':e,
    }
    
    net = ourNet(8, [32, 64, 96])
    
    list = net(batch)