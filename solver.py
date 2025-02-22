import os
import skimage.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LogWritter, calculate_mae
from data import generate_loader
from loss_fn import ConfidentLoss
from tqdm import tqdm
import datetime # new, show time on result(the pred_sal and pred_ctr)

class Solver():
    def __init__(self, module, opt):
        self.opt = opt
        self.logger = LogWritter(opt)
        
        self.dev = torch.device("cuda:{}".format(opt.GPU_ID) if torch.cuda.is_available() else "cpu")
        self.net = module.Net(opt)
        self.net = self.net.to(self.dev)
            
        msg = "# params:{}\n".format(sum(map(lambda x: x.numel(), self.net.parameters())))
        print(msg)
        self.logger.update_txt(msg=msg)

        self.loss_fn = ConfidentLoss(lmbd=opt.lmbda)
        
        # gather parameters
        base, head = [], []
        for name, param in self.net.named_parameters():
            if "backbone" in name:
                base.append(param)
            else:
                head.append(param)
        assert base!=[], 'backbone is empty'
        self.optim = torch.optim.Adam([{'params':base},{'params':head}], opt.lr,betas=(0.9, 0.999), eps=1e-8)

        self.train_loader = generate_loader("train", opt)
        self.eval_loader = generate_loader("test", opt)

        self.best_mae, self.best_step = 1, 0
        

    def fit(self):
        opt = self.opt
        
        for step in range(self.opt.max_epoch):
            # assign different learning rate
            power = (step+1)//opt.decay_step
            self.optim.param_groups[0]['lr'] = opt.lr * 0.1 * (0.5)**power   # for base
            self.optim.param_groups[1]['lr'] = opt.lr * (0.5)**power         # for head
            
            print('LR base: {}, LR head: {}'.format(self.optim.param_groups[0]['lr'],
                                                    self.optim.param_groups[1]['lr']))
            for i, inputs in enumerate(tqdm(self.train_loader)):

                self.optim.zero_grad()

                MASK = inputs[0].to(self.dev)
                IMG = inputs[1].to(self.dev)
                CTR = inputs[2].to(self.dev)

                pred = self.net(IMG)
                loss = self.loss_fn.get_value(pred, MASK, CTR)

                loss.backward()

                if opt.gclip > 0:
                    torch.nn.utils.clip_grad_value_(self.net.parameters(), opt.gclip)

                self.optim.step()
            # eval
            self.summary_and_save(step)
            

    def summary_and_save(self, step):
        print('evaluate...')
        mae = self.evaluate()

        if mae < self.best_mae:
            self.best_mae, self.best_step = mae, step+1
            self.save(step)
        else:
            if self.opt.save_every_ckpt:
                self.save(step)

        msg = "[{}/{}] {:.6f} (Best: {:.6f} @ {}K step)\n".format(step+1, self.opt.max_epoch,
                                                                  mae, self.best_mae,
                                                                  self.best_step)
        print(msg)
        self.logger.update_txt(msg=msg)

    @torch.no_grad()
    def evaluate(self):
        opt = self.opt
        self.net.eval()

        if opt.save_result:
            save_root = os.path.join(opt.save_root, opt.dataset)
            os.makedirs(save_root, exist_ok=True)

        mae = 0
        for i, inputs in enumerate(tqdm(self.eval_loader)):
            MASK = inputs[0].to(self.dev)
            IMG = inputs[1].to(self.dev)
            NAME = inputs[2][0]
            
            b,c,h,w = MASK.shape
            
            SOD = self.net(IMG)
            
            MASK = MASK.squeeze().detach().cpu().numpy()
            pred_sal, pred_ctr = SOD['sal'][-1], SOD['ctr'][-1]
            pred_sal = F.interpolate(pred_sal, (h, w), mode='bilinear', align_corners=False)
            pred_ctr = F.interpolate(pred_ctr, (h, w), mode='bilinear', align_corners=False)

            pred_sal = torch.sigmoid(pred_sal).squeeze().detach().cpu().numpy()
            pred_ctr = torch.sigmoid(pred_ctr).squeeze().detach().cpu().numpy()

            if opt.save_result:
                curr_time = datetime.datetime.now()       # add time in the name of result
                time_str = curr_time.strftime("%Y-%m-%d %H:%M")
                save_path_sal = os.path.join(save_root, "{}_sal_{}.png".format(NAME, time_str))
                save_path_ctr = os.path.join(save_root, "{}_ctr_{}.png".format(NAME, time_str))
                io.imsave(save_path_sal, pred_sal)
                io.imsave(save_path_ctr, pred_ctr)

            mae += calculate_mae(MASK, pred_sal)

        self.net.train()

        return mae/len(self.eval_loader)

    def load(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state_dict)
        
        return

    def save(self, step):
        os.makedirs(self.opt.ckpt_root, exist_ok=True)
        save_path = os.path.join(self.opt.ckpt_root, str(step)+".pt")
        torch.save(self.net.state_dict(), save_path)
