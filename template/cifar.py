# """
import copy
import os
import traceback
from datetime import datetime
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.backends import cudnn
from torch.optim.lr_scheduler import LinearLR

# from torchprofile import profile_macs

from template.model_block import MobileNetBottleneck, GhostBottleneck, Hswish

from pathlib import Path

import data_loader
import utils


class WylCNNModel(nn.Module):
    def __init__(self):
        super(WylCNNModel, self).__init__()
        self.Hswish = Hswish(inplace=True)
        self.drop_path_rate = 0.0
        # generated_init

        self.init_weight()

    def forward(self, x):
        # generate_forward

        out = self.avg(out)
        out = out.view(out.size(0), -1)

        out = F.dropout(out, p=0.2, training=self.training)
        out = self.fc(out)

        return out

    def forward_pre_gap(self, x):
        # generate_forward_pre_gap

        out = F.dropout(out, p=0.2, training=self.training)
        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='linear')

    def init_weight_gaussian(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    continue


class ScoreModel(object):
    def __init__(self):
        gpu = utils.GlobalConfigTool.get_gpu()
        self.device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
        net = WylCNNModel().to(self.device)
        self.net = net

        loss_function = nn.CrossEntropyLoss()
        self.loss_function = loss_function

        self.file_id = os.path.basename(__file__).split('.')[0]

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime('%Y-%m-%d %H:%M:%S')
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt' % (self.file_id), file_mode)
        f.write('[%s]-%s\n' % (dt, _str))
        f.flush()
        f.close()

    def compute_zen_score(self, mixup_gamma=1e-2, resolution=32, batch_size=96, repeat=32):
        info = {}
        nas_score_list = []

        with torch.no_grad():
            for repeat_count in range(repeat):
                self.net.init_weight_gaussian()
                input = torch.randn(size=[batch_size, 3, resolution, resolution], device=self.device)
                input2 = torch.randn(size=[batch_size, 3, resolution, resolution], device=self.device)
                mixup_input = input + mixup_gamma * input2
                output = self.net.forward_pre_gap(input)
                mixup_output = self.net.forward_pre_gap(mixup_input)

                nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
                nas_score = torch.mean(nas_score)

                # compute BN scaling
                log_bn_scaling_factor = 0.0
                for m in self.net.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                        log_bn_scaling_factor += torch.log(bn_scaling_factor)
                    pass
                pass
                nas_score = torch.log(nas_score) + log_bn_scaling_factor
                nas_score_list.append(float(nas_score))

        std_nas_score = np.std(nas_score_list)
        avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
        avg_nas_score = np.mean(nas_score_list)

        info['avg_nas_score'] = float(avg_nas_score)
        info['std_nas_score'] = float(std_nas_score)
        info['avg_precision'] = float(avg_precision)
        return info['avg_nas_score']

    def getgrad(self, model: torch.nn.Module, grad_dict: dict, step_iter=0):
        if step_iter == 0:
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                    # print(mod.weight.grad.data.size())
                    # print(mod.weight.data.size())
                    try:
                        grad_dict[name] = [mod.weight.grad.data.cpu().reshape(-1).numpy()]
                    except:
                        continue
        else:
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                    try:
                        grad_dict[name].append(mod.weight.grad.data.cpu().reshape(-1).numpy())
                    except:
                        continue
        return grad_dict


    # def caculate_zico(self, grad_dict, zico_weights):
    def caculate_zico(self, grad_dict):
        module_value, module_count = {}, {}
        for name, module in self.net.named_children():
            if name != 'Hswish' and name != 'avg':
                module_value[name] = 0.0
                module_count[name] = 0

        allgrad_array = None
        for i, modname in enumerate(grad_dict.keys()):
            grad_dict[modname] = np.array(grad_dict[modname])
        nsr_mean_sum = 0
        nsr_mean_sum_abs = 0
        nsr_mean_avg = 0
        nsr_mean_avg_abs = 0
        for j, modname in enumerate(grad_dict.keys()):
            nsr_std = np.std(grad_dict[modname], axis=0)
            nonzero_idx = np.nonzero(nsr_std)[0]
            nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
            tmpsum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
            if tmpsum == 0:
                pass
            else:
                block_name = modname.split('.')[0]
                module_value[block_name] += np.log(tmpsum)
                module_count[block_name] += 1

                # nsr_mean_sum_abs += np.log(tmpsum)
                # nsr_mean_avg_abs += np.log(np.mean(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx]))

        block_avg_sum_abs = {k: module_value[k] / module_count[k] for k in module_value}
        return sum(block_avg_sum_abs.values())

    def compute_zico_score(self, inputs, targets):
        grad_dict = {}
        self.net.train()
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        N = inputs.shape[0]
        split_data = 2

        for sp in range(split_data):
            st = sp * N // split_data
            en = (sp + 1) * N // split_data
            outputs = self.net.forward(inputs[st:en])
            loss = self.loss_function(outputs, targets[st:en])
            loss.backward()
            grad_dict = self.getgrad(self.net, grad_dict, sp)
        # print(grad_dict)
        res = self.caculate_zico(grad_dict)
        return res

    def compute_params(self):
        params = sum(p.numel() for p in self.net.parameters())
        return params / 1e6

    def compute_flops(self, image_size, channels):
        inputs = torch.randn(1, channels, image_size, image_size).to(self.device)
        flops = profile_macs(copy.deepcopy(self.net), inputs)
        # flops = 0.0
        return flops / 1e6


class TrainModel(object):
    def __init__(self, individual):
        data_root = utils.GlobalConfigTool.get_data_root()
        train_batch_size = utils.GlobalConfigTool.get_train_batch_size()
        num_workers = utils.GlobalConfigTool.get_num_workers()
        random_seed = utils.GlobalConfigTool.get_random_seed()
        # self.set_random_seed(random_seed)
        if utils.GlobalConfigTool.get_dataset() == "cifar10":
            full_train_loader = data_loader.Cifar10.get_train_loader(data_dir=data_root,
                                                                     batch_size=train_batch_size,
                                                                     augment=True,
                                                                     random_seed=None,
                                                                     shuffle=True,
                                                                     num_workers=num_workers,
                                                                     pin_memory=True)
            test_loader = data_loader.Cifar10.get_test_loader(data_dir=data_root,
                                                              batch_size=train_batch_size,
                                                              shuffle=False,
                                                              num_workers=num_workers,
                                                              pin_memory=True)
        elif utils.GlobalConfigTool.get_dataset() == "cifar100":
            full_train_loader = data_loader.Cifar100.get_train_loader(data_dir=data_root,
                                                                      batch_size=train_batch_size,
                                                                      augment=True,
                                                                      random_seed=None,
                                                                      shuffle=True,
                                                                      num_workers=num_workers,
                                                                      pin_memory=True)
            test_loader = data_loader.Cifar100.get_test_loader(data_dir=data_root,
                                                              batch_size=train_batch_size,
                                                              shuffle=False,
                                                              num_workers=num_workers,
                                                              pin_memory=True)
        else:
            pass
        self.full_train_loader = full_train_loader
        self.test_loader = test_loader

        gpu = utils.GlobalConfigTool.get_gpu()
        self.device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
        net = WylCNNModel().to(self.device)
        cudnn.benchmark = True
        loss_function = nn.CrossEntropyLoss()
        best_acc = 0.0

        self.net = net
        self.loss_function = loss_function
        self.best_acc = best_acc

        self.file_id = os.path.basename(__file__).split('.')[0]

        self.individual = copy.deepcopy(individual)

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime('%Y-%m-%d %H:%M:%S')
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt' % (self.file_id), file_mode)
        f.write('[%s]-%s\n' % (dt, _str))
        f.flush()
        f.close()

    def full_train(self, epoch, optimizer):
        self.net.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for _, data in enumerate(self.full_train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

        self.log_record('Train-Epoch:%4d, Loss: %.5f, Acc:%.5f' % (epoch + 1, running_loss / total, (correct / total)))

    def test(self, epoch):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for _, data in enumerate(self.test_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                loss = self.loss_function(outputs, labels)
                test_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.data).sum()
        if correct / total > self.best_acc:
            save_path = "./trained_models/best_CNN_" + str(self.file_id) + ".pth"
            torch.save(self.net.state_dict(), save_path)
            self.best_acc = correct / total
        self.log_record('Test-Epoch:%4d, Test-Loss:%.5f, Acc:%.5f' % (epoch + 1, test_loss / total, correct / total))

    def save_checkpoint(self, epoch, optimizer, scheduler):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': self.best_acc
        }
        save_path = "./trained_models/checkpoint_" + str(self.file_id) + ".pth"
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

    def load_checkpoint(self, optimizer, scheduler):
        checkpoint_path = "./trained_models/checkpoint_" + str(self.file_id) + ".pth"
        try:
            checkpoint = torch.load(checkpoint_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch'] + 1  # 下一次训练从这个 epoch 开始
            print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}")
            return start_epoch
        except FileNotFoundError:
            print("No checkpoint found, starting training from scratch.")
            return 0  # 从头开始

    def adjust_lr_warmup(self, ini_lr, lr_rate, optimizer, epoch):
        lr = ini_lr + ((lr_rate - ini_lr) / 5) * epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

    def process_test(self):
        total_epoch = utils.GlobalConfigTool.get_final_epoch()
        weight_decay = utils.GlobalConfigTool.get_weight_decay()
        drop_path_rate = utils.GlobalConfigTool.get_drop_path_rate()

        lr_rate = 0.05
        lr_rate_init = 0.0001
        optimizer = optim.SGD(self.net.parameters(), lr=lr_rate, momentum=0.9, weight_decay=weight_decay)
        warmup_scheduler = LinearLR(optimizer, start_factor=lr_rate_init / lr_rate, total_iters=5)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch - 5)

        start_epoch = self.load_checkpoint(optimizer, cosine_scheduler)

        for epoch in range(start_epoch, total_epoch):
            self.net.drop_path_rate = drop_path_rate * epoch / total_epoch
            if epoch < 5:
                self.full_train(epoch, optimizer)
                self.test(epoch)
                warmup_scheduler.step()
                # self.adjust_lr_warmup(lr_rate_init, lr_rate, optimizer_ini, epoch + 1)
            else:
                self.full_train(epoch, optimizer)
                self.test(epoch)
                cosine_scheduler.step()
                self.save_checkpoint(epoch, optimizer, cosine_scheduler)

        total_params = sum(p.numel() for p in self.net.parameters())
        return self.best_acc.item(), total_params / 1e6


class RunModel(object):
    def do_work(self, file_id, individual):
        m = TrainModel(individual)
        best_acc, params = 0.0, 0.0
        try:
            best_acc, params = m.process_test()
        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s' % (file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s' % (str(e)))
            traceback.print_exc()
        finally:
            m.log_record('Finished-Acc:%.5f' % best_acc)
            m.log_record('Params:%.5f' % params)

            return best_acc, params

    def score(self, inputs, labels, zen_repeat):
        m = ScoreModel()
        zen_score, zico_score = 0.0, 0.0

        try:
            # zen_score = m.compute_zen_score(resolution=32, batch_size=96, repeat=zen_repeat)
            zico_score = m.compute_zico_score(inputs, labels)
        except BaseException as e:
            traceback.print_exc()
        finally:
            # m.log_record('Zen_score:%.5f' % zen_score)
            # m.log_record('Zico_score:%.5f' % zico_score)
            return zen_score, zico_score

    def params_and_flops(self):
        m = ScoreModel()
        params, flops = 0.0, 0.0
        try:
            # zen_score = m.compute_zen_score(resolution=image_size, batch_size=batch_size)
            params = m.compute_params()
            # flops = m.compute_flops(image_size=32, channels=3)
        except BaseException as e:
            traceback.print_exc()
        finally:
            # m.log_record('Params:%.5f' % params)
            # m.log_record('Flops:%.5f' % flops)
            return params, flops
"""
