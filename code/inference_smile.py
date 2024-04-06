#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software;
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch
cuda = torch.cuda.is_available()
import torch
from allUtils.utils import  get_batch_unin_dataset_withlabel
import os
from codebase import utils as ut
import Causal_VAE_Smile as sup_dag
import argparse
cuda = torch.cuda.is_available()
from torchvision.utils import save_image
from minepy import MINE

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epoch_max',   type=int, default=101,    help="Number of training epochs")
parser.add_argument('--iter_save',   type=int, default=5, help="Save model every n epochs")
parser.add_argument('--run',         type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',       type=int, default=1,     help="Flag for training")
parser.add_argument('--color',       type=int, default=False,     help="Flag for color")
parser.add_argument('--toy',       type=str, default="smile_mask",     help="Flag for toy")
parser.add_argument('--dag',       type=str, default="sup_dag",     help="Flag for toy")

args = parser.parse_args()
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

layout = [
	('model={:s}',  'causalvae'),
	('run={:04d}', args.run),
  ('color=True', args.color),
  ('toy={:s}', str(args.toy))

]
model_name = '_'.join([t.format(v) for (t, v) in layout])
if args.dag == "sup_dag":
  lvae = sup_dag.CausalVAE(name=model_name, z_dim=16, inference=True).to(device)
  ut.load_BASE_by_name(lvae, 2)
  # ut.load_GNN_by_name(lvae, 9)
  # ut.load_SPN_by_name(lvae, 100)
  # ut.load_GNNSPN_by_name(lvae, 2)

# mask 3 epoch 45  mask 2 epoch 95  mask 1 epoch 100  mask 0 epoch 75
# mask 0 epoch 70 mask 1 epoch 100  mask 2 epoch 85  mask 3 epoch 80

if not os.path.exists('./figs_test_vae_smile/'):
    os.makedirs('./figs_test_vae_smile/')
means = torch.zeros(2, 3, 4).to(device)
z_mask = torch.zeros(2, 3, 4).to(device)

dataset_dir = '../causal_data/causal_data/smile'
train_dataset = get_batch_unin_dataset_withlabel(dataset_dir, 100, dataset="train")

count = 0
sample = False
print('DAG:{}'.format(lvae.dag.A))


mine = MINE(alpha=0.6, c=15)

label_cnt = torch.zeros((1, 400)).to(device)
dag_cnt = torch.zeros((1, 400)).to(device)

count_mic = 0
count_tic = 0

for u, l in train_dataset:
    # print('data', u.shape)
    for i in range(4):
        for j in range(-5, 5):
            re_label, re_dag, L, kl, rec, reconstructed_image, _ = lvae.negative_elbo_bound(u.to(device), l.to(device), mask=i, sample=sample, adj=j * 0)
        save_image(reconstructed_image[0], 'figs_test_vae_smile/base/reconstructed_image_{}_{}.png'.format(i, count), range=(0, 1))
    mine.compute_score(re_label.cpu().detach().numpy().reshape(400), re_dag.cpu().detach().numpy().reshape(400))
    mic = mine.mic()
    tic = mine.tic()
    count_mic += mic
    count_tic += tic
    label_cnt += re_label
    dag_cnt += re_dag
    save_image(u[0], './figs_test_vae_smile/base/true_{}.png'.format(count))
    count += 1
    if count == 10:
        break

label_avg = label_cnt / 40
dag_avg = dag_cnt / 40
print('label_dag\n')
print(label_avg)
print('\n')
print('dag_avg\n')
print(dag_avg)
print('\n')

print('mic\n')
print(count_mic/40)
print('\n')
print('tic\n')
print(count_tic/40)
print('\n')