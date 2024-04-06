# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software;
# you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch

cuda = torch.cuda.is_available()
import torch
from allUtils.utils import get_batch_unin_dataset_withlabel, _h_A

import os
from codebase import utils as ut
from Causal_VAE_Flow import CausalVAE
import argparse
from pprint import pprint

cuda = torch.cuda.is_available()
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import logging

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epoch_max', type=int, default=101, help="Number of training epochs")
parser.add_argument('--iter_save', type=int, default=5, help="Save model every n epochs")
parser.add_argument('--run', type=int, default=0, help="Run ID. In case you want to run replicates")
parser.add_argument('--train', type=int, default=1, help="Flag for training")
parser.add_argument('--color', type=int, default=False, help="Flag for color")
parser.add_argument('--toy', type=str, default="flow_mask", help="Flag for toy")
args = parser.parse_args()
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def _sigmoid(x):
    I = torch.eye(x.size()[0]).to(device)
    x = torch.inverse(I + torch.exp(-x))
    return x


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class DeterministicWarmup(object):
    """
    Linear deterministic warm-up as described in
    [S?nderby 2016].
    """

    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1 / n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t


layout = [
    ('model={:s}', 'causalvae'),
    ('run={:04d}', args.run),
    ('color=True', args.color),
    ('toy={:s}', str(args.toy))
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)
lvae = CausalVAE(name=model_name, z_dim=16).to(device)
if not os.path.exists(
        'figs_vae_flow/'):  # 判断所在目录下是否有该文件名的文件�?        os.makedirs('./logitdata_{}_{}/train/'.format(sample_num, context_dim))
    os.makedirs('figs_vae_flow/')

dataset_dir = '../causal_data/causal_data/flow_noise'
train_dataset = get_batch_unin_dataset_withlabel(dataset_dir, 64, "train")
test_dataset = get_batch_unin_dataset_withlabel(dataset_dir, 1, "test")
optimizer = torch.optim.Adam(lvae.parameters(), lr=1e-3, betas=(0.9, 0.999))
beta = DeterministicWarmup(n=100, t_max=1)  # Linear warm-up from 0 to 1 over 50 epoch
writer = SummaryWriter('./tensorboard/flow')


def save_model_by_name(model, global_step):
    save_dir = os.path.join('checkpoints_SPN', model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))


logger = get_logger('./loggers/model_flow.log')
logger.info('start training!')
for epoch in range(args.epoch_max):
    lvae.train()
    total_loss = 0
    total_rec = 0
    total_kl = 0
    for u, l in train_dataset:
        optimizer.zero_grad()
        u = u.to(device)
        re_label, re_dag, L, kl, rec, reconstructed_image, _ = lvae.negative_elbo_bound(u, l, sample=False)

        dag_param = lvae.dag.A

        h_a = _h_A(dag_param, dag_param.size()[0])
        L = L + 3 * h_a + 0.5 * h_a * h_a  # 对应论文中L=-ELBO+a*h_a+b*lu+y*lm

        L.backward()
        optimizer.step()
        total_loss += L.item()
        total_kl += kl.item()
        total_rec += rec.item()

        m = len(train_dataset)
        save_image(u[0], 'figs_vae_flow/reconstructed_image_true_{}.png'.format(epoch), normalize=True)

        save_image(reconstructed_image[0], 'figs_vae_flow/reconstructed_image_{}.png'.format(epoch), normalize=True)

    if epoch % 1 == 0:
        # print(str(epoch) + ' loss:' + str(total_loss / m) + ' kl:' + str(total_kl / m) + ' rec:' + str(
        #     total_rec / m) + 'm:' + str(m))
        logger.info('Epoch:[{}/{}]\t loss = {:.5f}\t kl = {:.5f}\t rec = {:.5f}\t'.format(epoch, args.epoch_max, total_loss / m, total_kl / m, total_rec / m))
        writer.add_scalars('add scalars', {'total_loss': total_loss/m, 'kl': total_kl/m, 'rec': total_rec/m}, epoch)

    if epoch % args.iter_save == 0:
        ut.save_GNNSPN_by_name(lvae, epoch)
        # ut.save_SPN_by_name(lvae, epoch)
        # ut.save_GNN_by_name(lvae, epoch)
        # ut.save_BASE_by_name(lvae, epoch)
        writer.add_image('build images', reconstructed_image[0], epoch, dataformats='CHW')
