# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software;
# you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import numpy as np
from codebase import utils as ut
from codebase import nns
from torch import nn

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


class CausalVAE(nn.Module):
    def __init__(self, nn='mask', name='vae', z_dim=16, z1_dim=4, z2_dim=4, inference=False):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.channel = 4
        self.scale = np.array([[20, 15], [2, 2], [59.5, 26.5], [10.5, 4.5]])

        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.channel)
        self.dec = nn.Decoder_DAG(self.z_dim, self.z1_dim, self.z2_dim)
        self.dag = nn.DagLayer(self.z1_dim, self.z1_dim, i=inference)
        self.attn = nn.Attention(self.z2_dim)
        self.mask_z = nn.MaskLayer(self.z_dim, concept=self.z1_dim)
        self.mask_u = nn.MaskLayer(self.z1_dim, concept=self.z1_dim, z1_dim=1)

        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

        self.weight1 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.weight2 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.weight3 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.weight4 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.weight5 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.weight6 = torch.nn.Parameter(torch.ones(1), requires_grad=True)

        self.weight7 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.weight8 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.weight9 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.weight10 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.weight11 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.weight12 = torch.nn.Parameter(torch.ones(1), requires_grad=True)

        self.w1 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.w4 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.w5 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.w6 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.w7 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.w8 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.w9 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.w10 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.w11 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.w12 = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def negative_elbo_bound(self, x, label, mask=None, sample=False, adj=None, lambdav=0.001):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """

        assert label.size()[1] == self.z1_dim  # 确保label.size()[1]等于self.z1_dim，否则报异常

        re_label = None
        re_dag = None

        # x[64, 4, 96, 96], label[64, 4]
        q_m, q_v = self.enc.encode(x.to(device))  # 均值，方差

        # q_m[batch_size, 4, 4], q_v[batch_size, 4, 4]
        q_m, q_v = q_m.reshape([q_m.size()[0], self.z1_dim, self.z2_dim]), torch.ones(q_m.size()[0], self.z1_dim,
                                                                                      self.z2_dim).to(device)

        q_m = self.enc.dag_gnn(q_m)
        # 0gender, 1smile, 2narrow eyes, 3mouth slightly open
        # 0gender → 2narrow eyes, 1smile → (2narrow eyes, 3mouth open), 3mouth open → 2narrow eyes

        # 编码过程 x->z z=A^T*Z+e，即z = (I-A^T)^(-1)*e, e∈N[0,1]
        decode_m, decode_v = self.dag.calculate_dag(q_m.to(device),
                                                    torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(device))
        decode_m, decode_v = decode_m.reshape([q_m.size()[0], self.z1_dim, self.z2_dim]), decode_v

        e1, e2, e3, e4 = decode_m[:, 0, :], decode_m[:, 1, :], decode_m[:, 2, :], decode_m[:, 3, :]

        e3 = self.weight1 * e1 + self.weight2 * e2 + self.weight3 * e4
        e4 = self.weight4 * e2

        decode_m = torch.cat((e1.reshape([-1, 1, 4]), e2.reshape([-1, 1, 4]), e3.reshape([-1, 1, 4]),
                              e4.reshape([-1, 1, 4])), dim=1)

        if sample is False:
            if mask is not None and mask in [0, 1, 3]:
                z_mask = torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(device) * adj
                decode_m[:, mask, :] = z_mask[:, mask, :]  # 第mask行设为adj（mask干预原因）
                decode_v[:, mask, :] = z_mask[:, mask, :]
            # 第一次mask后，z*因果矩阵  z = A^T * z  即m_zm = A^T * decode_m
            m_zm, m_zv = self.dag.mask_z(decode_m.to(device)).reshape(
                [q_m.size()[0], self.z1_dim, self.z2_dim]), decode_v.reshape([q_m.size()[0], self.z1_dim, self.z2_dim])
            # m_zm [64, 4, 4]
            # print('m_zm\n', m_zm.shape)

            # 第一次mask之后，label*因果矩阵，即解耦后的表征   m_u = A^T * label
            m_u = self.dag.mask_u(label.to(device))

            # mask
            # [100, 4, 4]
            f_z = self.mask_z.mix(m_zm).reshape([q_m.size()[0], self.z1_dim, self.z2_dim]).to(device)

            e_tilde = self.attn.attention(decode_m.reshape([q_m.size()[0], self.z1_dim, self.z2_dim]).to(device),
                                          q_m.reshape([q_m.size()[0], self.z1_dim, self.z2_dim]).to(device))[0]

            # print('e_tilde\n', e_tilde.shape)

            f_z1 = f_z + e_tilde
            if mask is not None and mask == 2:
                z_mask = torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(device) * adj
                f_z1[:, mask, :] = z_mask[:, mask, :]  # 第mask行设为adj（mask干预结果，不会改变原因）
                m_zv[:, mask, :] = z_mask[:, mask, :]
            g_u = self.mask_u.mix(m_u).to(device)

            m_zv = torch.ones([q_m.size()[0], self.z1_dim, self.z2_dim]).to(device)
            # 待解码的z = f_z1+根号(q_v * lambdav)*sample
            z_given_dag = ut.conditional_sample_gaussian(f_z1, q_v * lambdav)

            z_given_dag_ave = self.ave(z_given_dag)

            if mask is not None:
                re_label = label.reshape(1, 400)
                re_dag = z_given_dag_ave.reshape(1, 400)

            # if mask != None:
            #     print('z_given_dag\n', z_given_dag.shape)
            #     print('u\n', label.reshape(1, 400))
            #     print('z_given_dag\n', z_given_dag_ave.reshape(1, 400))
            # z_given_dag [64, 4, 4]

            z1, z2, z3, z4 = z_given_dag[:, 0, :], z_given_dag[:, 1, :], z_given_dag[:, 2, :], z_given_dag[:, 3, :]
            with torch.no_grad():
                z3 = self.weight1 * z1 + self.weight2 * z2 + self.weight3 * z4
                z4 = self.weight4 * z2

            z_given_dag = torch.cat(
                (z1.reshape([-1, 1, 4]), z2.reshape([-1, 1, 4]), z3.reshape([-1, 1, 4]), z4.reshape([-1, 1, 4])), dim=1)

        decoded_bernoulli_logits, x1, x2, x3, x4 = self.dec.decode_sep(
            z_given_dag.reshape([z_given_dag.size()[0], self.z_dim]), label.to(device))
        # decoded_bernoulli_logits [64, 4*96*96]

        rec = ut.log_bernoulli_with_logits(x, decoded_bernoulli_logits.reshape(x.size()))  # 重构损失，使用交叉熵损失计算
        rec = -torch.mean(rec)

        p_m, p_v = torch.zeros(q_m.size()), torch.ones(q_m.size())
        cp_m, cp_v = ut.condition_prior(self.scale, label, self.z2_dim)  # 均值，方差

        cp_v = torch.ones([q_m.size()[0], self.z1_dim, self.z2_dim]).to(device)
        cp_z = ut.conditional_sample_gaussian(cp_m.to(device), cp_v.to(device))
        kl = torch.zeros(1).to(device)
        # qm = mu, qv = sigma2 , pm = 0 , pv = 1
        kl = 0.3 * ut.kl_normal(q_m.view(-1, self.z_dim).to(device), q_v.view(-1, self.z_dim).to(device),
                                p_m.view(-1, self.z_dim).to(device), p_v.view(-1, self.z_dim).to(device))

        for i in range(self.z1_dim):
            # ut.kl_normal是KL散度计算  decode_m是因果图  cp_v = 1  cp_m是均值
            kl = kl + 1 * ut.kl_normal(decode_m[:, i, :].to(device), cp_v[:, i, :].to(device), cp_m[:, i, :].to(device),
                                       cp_v[:, i, :].to(device))
        kl = torch.mean(kl)
        mask_kl = torch.zeros(1).to(device)  # mask_kl = 0
        mask_kl2 = torch.zeros(1).to(device)
        for i in range(self.z1_dim):
            # ut.kl_normal是KL散度计算  f_z1 = f_z+e_tilde  cp_v = 1  cp_m是均值
            mask_kl = mask_kl + 1 * ut.kl_normal(f_z1[:, i, :].to(device), cp_v[:, i, :].to(device),
                                                 cp_m[:, i, :].to(device), cp_v[:, i, :].to(device))
        u_loss = torch.nn.MSELoss()
        # cos_loss = nn.HingeEmbeddingLoss(margin=0.2)
        mask_l = torch.mean(mask_kl) + u_loss(g_u, label.float().to(device))
        nelbo = rec + kl + mask_l
        return re_label, re_dag, nelbo, kl, rec, decoded_bernoulli_logits.reshape(x.size()), z_given_dag

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))

    def ave(self, z):
        z = torch.sum(z, dim=-1) / 4
        z_ave = z.reshape([z.size()[0], 4])
        return z_ave
