import os
import torch
from exp.exp_basic import Exp_Basic
from models.model_factory import get_model
from data_provider.data_factory import get_data
from utils.loss import L2Loss
from utils.visual import visual
import matplotlib.pyplot as plt
import numpy as np
import math


class Exp_Steady(Exp_Basic):

    def __init__(self, args):
        super(Exp_Steady, self).__init__(args)

    def vali(self):
        myloss = L2Loss(size_average=False)
        self.model.eval()
        rel_err = 0.0
        with torch.no_grad():
            for pos, fx, cond, y in self.test_loader:
                x, fx, cond, y = pos.cuda(), fx.cuda(), cond.cuda(), y.cuda()
                fx = torch.cat((fx, cond.repeat(1, fx.shape[1], 1)), dim=-1)
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x[:, :, :3], fx)
                if self.args.normalize:
                    out = self.dataset.y_normalizer.decode(out)

                tl = myloss(out, y).item()
                rel_err += tl

        rel_err /= self.args.ntest
        return rel_err

    def train(self):
        ### load GeoPT pre-trained model
        if self.args.finetune:
            self.model = self.load_pretrained_with_filter(self.model,
                                                          "./checkpoints/" + self.args.finetune_name + ".pt")
        if self.args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise ValueError('Optimizer only AdamW or Adam')

        ### adopt learning rate scheduler
        if self.args.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.lr, epochs=self.args.epochs,
                                                            steps_per_epoch=len(self.train_loader),
                                                            pct_start=self.args.pct_start)
        elif self.args.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        elif self.args.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        myloss = L2Loss(size_average=False)

        train_loss_list = []
        test_loss_list = []

        for ep in range(self.args.epochs):

            self.model.train()
            train_loss = 0

            for pos, fx, cond, y in self.train_loader:
                x, fx, cond, y = pos.cuda(), fx.cuda(), cond.cuda(), y.cuda()
                fx = torch.cat((fx, cond.repeat(1, fx.shape[1], 1)), dim=-1)
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x[:, :, :3], fx)
                if self.args.normalize:
                    out = self.dataset.y_normalizer.decode(out)
                    y = self.dataset.y_normalizer.decode(y)
                loss = myloss(out, y)

                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()

                if self.args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()

                if self.args.scheduler == 'OneCycleLR':
                    scheduler.step()
            if self.args.scheduler == 'CosineAnnealingLR' or self.args.scheduler == 'StepLR':
                scheduler.step()

            train_loss = train_loss / self.args.ntrain
            print("Epoch {} Train loss : {:.5f}".format(ep, train_loss))
            train_loss_list.append(train_loss)

            rel_err = self.vali()
            print("rel_err:{}".format(rel_err))
            test_loss_list.append(rel_err)

            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save models')
                torch.save(self.model.state_dict(), os.path.join('./checkpoints', self.args.save_name + '.pt'))

            if ep % 10 == 0:
                if not os.path.exists('./training_logs'):
                    os.makedirs('./training_logs')
                print('save logs')
                np.save(os.path.join('./training_logs', self.args.save_name + '_train_loss.npy'),
                        np.array(train_loss_list))
                np.save(os.path.join('./training_logs', self.args.save_name + '_test_loss.npy'),
                        np.array(test_loss_list))

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('final save models')
        torch.save(self.model.state_dict(), os.path.join('./checkpoints', self.args.save_name + '.pt'))
        if not os.path.exists('./training_logs'):
            os.makedirs('./training_logs')
        print('final training logs')
        np.save(os.path.join('./training_logs', self.args.save_name + '_train_loss.npy'), np.array(train_loss_list))
        np.save(os.path.join('./training_logs', self.args.save_name + '_test_loss.npy'), np.array(test_loss_list))

    def test(self):
        self.model.load_state_dict(torch.load("./checkpoints/" + self.args.save_name + ".pt"))
        self.model.eval()
        if not os.path.exists('./results/' + self.args.save_name + '/'):
            os.makedirs('./results/' + self.args.save_name + '/')

        rel_err = 0.0
        rel_err_split = 0.0
        rel_err_split_max = 0.0
        id = 0
        mse = 0.0
        mae = 0.0
        myloss = L2Loss(size_average=False)

        with torch.no_grad():
            for pos, fx, cond, y in self.test_loader:
                id += 1
                x, fx, cond, y = pos.cuda(), fx.cuda(), cond.cuda(), y.cuda()
                fx = torch.cat((fx, cond.repeat(1, fx.shape[1], 1)), dim=-1)
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x[:, :, :3], fx)
                if self.args.normalize:
                    out = self.dataset.y_normalizer.decode(out)
                tl = myloss(out, y).item()
                mse += (out - y).pow(2).mean(dim=1).mean(dim=1).sum().item()
                mae += torch.abs(out - y).mean(dim=1).mean(dim=1).sum().item()
                rel_err += tl
                rel_err_split += torch.mean(torch.mean(torch.abs(out - y) / torch.abs(y), dim=0), dim=0)
                rel_err_split_max += torch.max(torch.max(torch.abs(out - y) / torch.abs(y), dim=0)[0], dim=0)[0]
                if id < self.args.vis_num:
                    print('visual: ', id)
                    visual(x, y, out, self.args, id)

        rel_err /= self.args.ntest
        mse /= self.args.ntest
        mae /= self.args.ntest
        rel_err_split /= self.args.ntest
        rel_err_split_max /= self.args.ntest
        print("test rel_err:{}".format(rel_err))
        print("test mse:{}".format(mse))
        print("test mae:{}".format(mae))
        print("test rel_err split:{}".format(rel_err_split))
        print("test rel_err split max:{}".format(rel_err_split_max))

    def test_full_mesh(self):
        self.model.load_state_dict(torch.load("./checkpoints/" + self.args.save_name + ".pt"))
        self.model.eval()
        if not os.path.exists('./results/' + self.args.save_name + '/'):
            os.makedirs('./results/' + self.args.save_name + '/')

        rel_err = 0.0
        rel_err_split = 0.0
        rel_err_split_max = 0.0
        id = 0
        mse = 0.0
        mae = 0.0
        myloss = L2Loss(size_average=False)

        with torch.no_grad():
            for pos, fx, cond, y in self.test_loader_full:
                id += 1
                x, fx, cond, y = pos.cuda(), fx.cuda(), cond.cuda(), y.cuda()
                fx = torch.cat((fx, cond.repeat(1, fx.shape[1], 1)), dim=-1)
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x[:, :, :3], fx)
                if self.args.normalize:
                    out = self.dataset.y_normalizer.decode(out)
                tl = myloss(out, y).item()
                mse += (out - y).pow(2).mean(dim=1).mean(dim=1).sum().item()
                mae += torch.abs(out - y).mean(dim=1).mean(dim=1).sum().item()
                rel_err += tl
                rel_err_split += torch.mean(torch.mean(torch.abs(out - y) / torch.abs(y), dim=0), dim=0)
                rel_err_split_max += torch.max(torch.max(torch.abs(out - y) / torch.abs(y), dim=0)[0], dim=0)[0]
                if id < self.args.vis_num:
                    print('visual: ', id)
                    visual(x, y, out, self.args, id)

        rel_err /= self.args.ntest
        mse /= self.args.ntest
        mae /= self.args.ntest
        rel_err_split /= self.args.ntest
        rel_err_split_max /= self.args.ntest
        print("test rel_err:{}".format(rel_err))
        print("test mse:{}".format(mse))
        print("test mae:{}".format(mae))
        print("test rel_err split:{}".format(rel_err_split))
        print("test rel_err split max:{}".format(rel_err_split_max))
