"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import copy

import torch
import torch.nn as nn

from networks import FewShotGen, GPPatchMcResDis


def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))


class FUNITModel(nn.Module):
    def __init__(self, hp):
        super(FUNITModel, self).__init__()
        self.gen = FewShotGen(hp['gen'])
        self.dis = GPPatchMcResDis(hp['dis'])
        self.gen_test = copy.deepcopy(self.gen)

    def forward(self, co_data, cl_data, hp, mode):
        """
        Params:
            - co_data: content data with one content image and one content image label
            - cl_data: class data with one class image and one class image label
            - hp: hyperparameters, more specifically weights for the losses
            - mode: discriminator or generator update step
        Returns:
            Generator:
                - l_total: overall generator loss
                - l_adv: adversarial loss of generator
                - l_x_rec: reconstruction loss
                - l_c_rec: feature matching loss same image
                - l_m_rec: feature matching loss translated image
                - acc: generator accuracy
            Discriminator:
                - l_total: overall discriminator loss
                - l_fake_p: discriminator fake loss
                - l_real_pre: discriminator real loss
                - l_reg_pre: real gradient penalty regularization loss term
                - acc: discriminator accuracy
        """
        xa = co_data[0].cuda()
        la = co_data[1].cuda()
        xb = cl_data[0].cuda()
        lb = cl_data[1].cuda()

        if mode == 'gen_update':

            # forward pass
            c_xa = self.gen.enc_content(xa)
            s_xa = self.gen.enc_class_model(xa)
            s_xb = self.gen.enc_class_model(xb)
            xt = self.gen.decode(c_xa, s_xb)  # translation
            xr = self.gen.decode(c_xa, s_xa)  # reconstruction

            # adversarial loss, generator accuracy and features
            l_adv_t, gacc_t, xt_gan_feat = self.dis.calc_gen_loss(xt, lb) # calc_gen_loss returns loss, accuracy and gan_feat of only first param, i.e. xt
            l_adv_r, gacc_r, xr_gan_feat = self.dis.calc_gen_loss(xr, la)

            # extracting features for the feature matching loss
            _, xb_gan_feat = self.dis(xb, lb) 
            _, xa_gan_feat = self.dis(xa, la) 

            # feature matching loss
            l_c_rec = recon_criterion(xr_gan_feat.mean(3).mean(2), xa_gan_feat.mean(3).mean(2))
            l_m_rec = recon_criterion(xt_gan_feat.mean(3).mean(2), xb_gan_feat.mean(3).mean(2))
            
            # reconstruction loss
            l_x_rec = recon_criterion(xr, xa)

            # adversarial loss for 
            l_adv = 0.5 * (l_adv_t + l_adv_r)

            # accuracy
            acc = 0.5 * (gacc_t + gacc_r)

            # overall loss: adversarial, reconstruction and feature matching loss
            l_total = (hp['gan_w'] * l_adv + hp['r_w'] * l_x_rec + hp['fm_w'] * (l_c_rec + l_m_rec))
            l_total.backward()
            return l_total, l_adv, l_x_rec, l_c_rec, l_m_rec, acc

        elif mode == 'dis_update':
            xb.requires_grad_()

            # calculate discriminator's real loss
            l_real_pre, acc_r, resp_r = self.dis.calc_dis_real_loss(xb, lb)
            l_real = hp['gan_w'] * l_real_pre
            l_real.backward(retain_graph=True)

            # real gradient penalty regularization proposed by Mescheder et al.
            l_reg_pre = self.dis.calc_grad2(resp_r, xb)
            l_reg = 10 * l_reg_pre
            l_reg.backward()

            # generate images for the discriminator to classify
            with torch.no_grad():
                c_xa = self.gen.enc_content(xa)
                s_xb = self.gen.enc_class_model(xb)
                xt = self.gen.decode(c_xa, s_xb)

            # calculate discriminator's fake loss
            l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(xt.detach(), lb)
            l_fake = hp['gan_w'] * l_fake_p
            l_fake.backward()
            l_total = l_fake + l_real + l_reg

            acc = 0.5 * (acc_f + acc_r)
            return l_total, l_fake_p, l_real_pre, l_reg_pre, acc
        else:
            assert 0, 'Not support operation'

    def test(self, co_data, cl_data):
        """
        Params:
            - co_data: content data with one content image and one content image label
            - cl_data: class data with one class image and one class image label
        """          
        self.eval()
        self.gen.eval()
        self.gen_test.eval()

        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()

        c_xa_current = self.gen.enc_content(xa)
        s_xa_current = self.gen.enc_class_model(xa)
        s_xb_current = self.gen.enc_class_model(xb)
        xt_current = self.gen.decode(c_xa_current, s_xb_current)
        xr_current = self.gen.decode(c_xa_current, s_xa_current)

        c_xa = self.gen_test.enc_content(xa)
        s_xa = self.gen_test.enc_class_model(xa)
        s_xb = self.gen_test.enc_class_model(xb)
        xt = self.gen_test.decode(c_xa, s_xb)
        xr = self.gen_test.decode(c_xa, s_xa)

        self.train()
        
        return xa, xr_current, xt_current, xb, xr, xt

    def translate_k_shot(self, co_data, cl_data, k):
        """
        Params:
            - co_data: content data with one content image and one content image label
            - cl_data: class data with one class image and one class image label
            - k: number of shots to generate a translated image
        """          
        self.eval()

        # for training on GPU
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()

        c_xa_current = self.gen_test.enc_content(xa)

        # perform translation for k shots
        if k == 1:
            c_xa_current = self.gen_test.enc_content(xa)
            s_xb_current = self.gen_test.enc_class_model(xb)
            xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        else:
            s_xb_current_before = self.gen_test.enc_class_model(xb)
            s_xb_current_after = s_xb_current_before.squeeze(-1).permute(1,
                                                                         2,
                                                                         0)
            s_xb_current_pool = torch.nn.functional.avg_pool1d(
                s_xb_current_after, k)
            s_xb_current = s_xb_current_pool.permute(2, 0, 1).unsqueeze(-1)
            xt_current = self.gen_test.decode(c_xa_current, s_xb_current)

        return xt_current

    def compute_k_style(self, style_batch, k):
        self.eval()
        style_batch = style_batch.cuda()
        s_xb_before = self.gen_test.enc_class_model(style_batch)
        s_xb_after = s_xb_before.squeeze(-1).permute(1, 2, 0)
        s_xb_pool = torch.nn.functional.avg_pool1d(s_xb_after, k)
        s_xb = s_xb_pool.permute(2, 0, 1).unsqueeze(-1)
        return s_xb

    def translate_simple(self, content_image, class_code):
        self.eval()
        xa = content_image.cuda()
        s_xb_current = class_code.cuda()
        c_xa_current = self.gen_test.enc_content(xa)
        xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        return xt_current
