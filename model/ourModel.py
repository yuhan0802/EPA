import os
from torch.optim import AdamW
from common.size_adapter import BatchSizeAdapter
from model.netGenerator import netGenerator
from model.loss import *
from common.laplacian import *
import lpips
import torch.nn.functional as F
from model.ourNet import ourNet
from util.utils_func import get_z


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, args):
        self.args = args
        cond_c = [32, 64, 96]
        self.my_net = ourNet(args.bins, cond_c)
        self.device()
        self.is_multiple()
        self.optimG = AdamW(self.my_net.parameters(), lr=1e-6, weight_decay=1e-3)
        self.is_resume()
        self.lap = LapLoss()
        self.l2 = nn.MSELoss()
        self.perc = lpips.LPIPS(net='alex').to(device)
        self.batch_size_adapter = BatchSizeAdapter()
        self.get_parameter_number()

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.my_net.parameters())
        trainable_num = sum(p.numel() for p in self.my_net.parameters() if p.requires_grad)
        print('Total parameter: ', total_num/1e6, ', Trainable: ', trainable_num/1e6)

    def is_multiple(self):
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print("Start with", num_gpus, "GPUs!")
            self.my_net = nn.DataParallel(self.my_net, device_ids=list(range(num_gpus)))

    def is_resume(self):
        if self.args.RESUME:
            print("Start from ", self.args.RESUME_EPOCH)
            if self.args.RESUME_EPOCH % 5 == 0:
                path_checkpoint = "train/checkpoint/train_ckpt_" + str(self.args.RESUME_EPOCH) + ".pth"
            else:
                path_checkpoint = "train/checkpoint/train_ckpt.pth"

            weights = {
                k.replace("module.", ""): v for k, v in (torch.load(path_checkpoint)['net']).items()
            }
            self.my_net.load_state_dict(weights)
            self.optimG.load_state_dict(torch.load(path_checkpoint)['optimizer'])

    def train(self):
        self.my_net.train()

    def eval(self):
        self.my_net.eval()

    def device(self):
        self.my_net.to(device)

    def load_model(self, path):
        weights = {
            k.replace("module.", ""): v for k, v in (torch.load(path, weights_only=False)['net']).items()
        }
        self.my_net.load_state_dict(weights)

    def save_model(self, path):
        torch.save(self.my_net.state_dict(), '{}/work.pkl'.format(path))

    def save_model_min_loss(self, path):
        torch.save(self.my_net.state_dict(), '{}/min_loss.pkl'.format(path))

    def save_checkpoint(self, type, epoch):
        checkpoint = {
            "net": self.my_net.state_dict(),
            'optimizer': self.optimG.state_dict(),
            "epoch": epoch
        }
        if hasattr(self.args, 'dataset_finetune'):
            finetune_name = self.args.dataset_finetune
        else:
            finetune_name = "train"

        if not os.path.isdir("./train/checkpoint"):
            os.mkdir("./train/checkpoint")
        if type == 'everyone':
            torch.save(checkpoint, './train/checkpoint/' + finetune_name + '_ckpt.pth')
        else:
            torch.save(checkpoint, './train/checkpoint/' + finetune_name + '_ckpt_%s_perc_l1loss.pth' % (str(epoch)))

    def inverse_normalize(self, tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        tensor = tensor * std[:, None, None] + mean[:, None, None]
        return tensor
    
    def inference(self, batch, heat=0.3):
        batch = self.batch_size_adapter.pad(batch)
        self.eval()
        
        _, _, pred, _, _ = self.my_net(batch)
        pred = self.inverse_normalize(pred)
        pred = torch.clamp(pred, 0, 1)

        pred = self.batch_size_adapter.unpad(pred)
        return pred

    def update(self, batch, learning_rate):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        self.train()
        pure_gt = batch['pure_gt']

        Gts, Gts_pred, pred, Gts_pred_0, Gts_pred_1 = self.my_net(batch)
        pred = self.inverse_normalize(pred)
        pred = torch.clamp(pred, 0, 1)

        loss = self.perc(pred, pure_gt)
        # for fea_gt, fea_pred, p0, p1 in zip(Gts, Gts_pred, Gts_pred_0, Gts_pred_1):
        #     loss += self.l2(fea_gt, fea_pred)
        for fea_gt, fea_pred, p0, p1 in zip(Gts, Gts_pred, Gts_pred_0, Gts_pred_1):
            loss += self.l2(fea_gt, fea_pred) + 0.5 * self.l2(fea_gt, p0) + 0.1 * self.l2(fea_gt, p1)
        # for fea_gt, fea_pred, p0, p1 in zip(Gts, Gts_pred, Gts_pred_0, Gts_pred_1):
        #     loss += F.l1_loss(fea_gt, fea_pred) + 0.1 * F.l1_loss(fea_gt, p0) + 0.1 * F.l1_loss(fea_gt, p1)
        loss = loss.mean()
        self.optimG.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.my_net.parameters(), 1)
        self.optimG.step()

        return pred, loss