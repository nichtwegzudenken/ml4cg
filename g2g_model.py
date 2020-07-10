import copy 

import torch
import torch.nn as nn

from networks import FewShotGen, GPPatchMcResDis

def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))

class G2GModel(nn.Module):

    def __init__(self, hp):
        super(G2GModel, self).__init__()
        self.gen = FewShotGen(hp['gen'])
        self.dis = GPPatchMcResDis(hp['dis'])
        self.gen_test = copy.deepcopy(self.gen)

    def cross_forward(self, x1, x2, mode):
        if mode == 'train':
            gen = self.gen 
        else:
            gen = self.gen_test

        # content and class codes image x1 (dog)
        x1_a = gen.enc_content(x1)
        x1_b = gen.enc_class_model(x1)

        # content and class codes image x2 (meerkat)
        x2_a = gen.enc_content(x2)
        x2_b = gen.enc_class_model(x2)

        # mixed images
        m1 = gen.decode(x1_a, x2_b)  # translation a,b meerkat
        m2 = gen.decode(x2_a, x1_b) # translation b,a dog

        # reconstruction with itself
        xr_a = gen.decode(x1_a, x1_b)  # reconstruction of image a (dog)
        xr_b = gen.decode(x2_a, x2_b) # reconstruction of image b (meerkat)
        
        # content and class codes of mixed image m1
        m1_a = gen.enc_content(m1) 
        m1_b = gen.enc_class_model(m1)

        # content and class codes of mixed image m2
        m2_a = gen.enc_content(m2)
        m2_b = gen.enc_class_model(m2)

        # reconstruction after reassembly stage
        r1 = gen.decode(m1_a, m2_b)
        r2 = gen.decode(m2_a, m1_b)

        return {
            "x1_a": x1_a,
            "x1_b": x1_b,
            "x2_a": x2_a,
            "x2_b": x2_b,
            "m1": m1,
            "m2": m2,
            "xr_a": xr_a,
            "xr_b": xr_b,
            "m1_a": m1_a,
            "m1_b": m1_b,
            "m2_a": m2_a,
            "m2_b": m2_b,
            "r1": r1,
            "r2": r2,
        }

    def calc_g_loss(self, out, xa, xb, la, lb):
        # features for two different images: the mixed image and the original of the same class
        l_adv_t_b, gacc_t_b, xt_b_gan_feat = self.dis.calc_gen_loss(out["m1"], lb) # calc_gen_loss returns loss, accuracy and gan_feat of only first param, i.e. xt_a
        l_adv_t_a, gacc_t_a, xt_a_gan_feat = self.dis.calc_gen_loss(out["m2"], la) # calc_gen_loss returns loss, accuracy and gan_feat of only first param, i.e. xt_a

        # features for two same images: the reconstructed image and the original
        l_adv_r_a, gacc_r_a, xr_a_gan_feat = self.dis.calc_gen_loss(out["xr_a"], la) # calc_gen_loss returns loss, accuracy and gan_feat of only first param, i.e. xt_a
        l_adv_r_b, gacc_r_b, xr_b_gan_feat = self.dis.calc_gen_loss(out["xr_b"], lb) # calc_gen_loss returns loss, accuracy and gan_feat of only first param, i.e. xt_a

        # extracting features for the feature matching loss
        _, xb_gan_feat = self.dis(xb, lb) # this extracts only features of xb
        _, xa_gan_feat = self.dis(xa, la) # this extracts only features of xa

        # feature matching loss between the two same images: the reconstructed image and the original
        l_c_rec_a = recon_criterion(xr_a_gan_feat.mean(3).mean(2), xa_gan_feat.mean(3).mean(2))
        l_c_rec_b = recon_criterion(xr_b_gan_feat.mean(3).mean(2), xb_gan_feat.mean(3).mean(2))
        l_c_rec = 0.5 * (l_c_rec_a + l_c_rec_b)

        # feature matching loss between two different images: the mixed image and the original of the same class
        l_m_rec_a = recon_criterion(xt_a_gan_feat.mean(3).mean(2), xa_gan_feat.mean(3).mean(2))
        l_m_rec_b = recon_criterion(xt_b_gan_feat.mean(3).mean(2), xb_gan_feat.mean(3).mean(2))
        l_m_rec = 0.5 * (l_m_rec_a + l_m_rec_b)

        # reconstruction loss
        l_x_rec = recon_criterion(xa, out["xr_a"])
        l_x_rec = recon_criterion(xb, out["xr_b"])

        # adversarial loss for 
        l_adv = 0.25 * (l_adv_t_a + l_adv_t_b + l_adv_r_a + l_adv_r_b)

        # accuracy
        acc = 0.25 * (gacc_t_a + gacc_t_b + gacc_r_a + gacc_r_b)

        return l_adv, l_x_rec, l_c_rec, l_m_rec, acc

    def calc_d_loss(self, xa, xb, la, lb):
        # calculate discriminator's real loss
        l_real_pre_a, acc_r_a, resp_r_a = self.dis.calc_dis_real_loss(xa, la)
        l_real_pre_b, acc_r_b, resp_r_b = self.dis.calc_dis_real_loss(xb, lb)
        l_real = (l_real_pre_a + l_real_pre_b)
        l_real.backward(retain_graph=True)

        # real gradient penalty regularization proposed by Mescheder et al.
        l_reg_pre_a = self.dis.calc_grad2(resp_r_a, xa)
        l_reg_pre_b = self.dis.calc_grad2(resp_r_b, xb)
        l_reg = (l_reg_pre_a + l_reg_pre_b)
        l_reg.backward()

        # generate images for the discriminator to classify
        with torch.no_grad():
            c_xa = self.gen.enc_content(xa)
            c_xb = self.gen.enc_content(xb)
            
            s_xa = self.gen.enc_class_model(xa)
            s_xb = self.gen.enc_class_model(xb)

            xt_a = self.gen.decode(c_xa, s_xb) # meerkat 
            xt_b = self.gen.decode(c_xb, s_xa) # dog

        # calculate discriminator's fake loss
        l_fake_p_a, acc_f_a, resp_f_a = self.dis.calc_dis_fake_loss(xt_a.detach(), lb)
        l_fake_p_b, acc_f_b, resp_f_b = self.dis.calc_dis_fake_loss(xt_b.detach(), la)

        l_fake = (l_fake_p_a + l_fake_p_b)
        l_fake.backward()

        acc = 0.25 * (acc_f_a + acc_f_b + acc_r_a + acc_r_b)        
        return l_fake, l_real, l_reg, acc

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
            out = self.cross_forward(xa, xb, 'train')

            l_adv, l_x_rec, l_c_rec, l_m_rec, acc = self.calc_g_loss(out, xa, xb, la, lb)            
            l_total.backward()

            # overall loss: adversarial, reconstruction and feature matching loss
            l_total = (hp['gan_w'] * l_adv + hp['r_w'] * l_x_rec + hp['fm_w'] * (l_c_rec + l_m_rec))

            return l_total, l_adv, l_x_rec, l_c_rec, l_m_rec, acc

        elif mode == 'dis_update':
            xb.requires_grad_()

            l_fake, l_real, l_reg, acc = self.calc_d_loss(xa, xb, la, lb)

            # overall loss: fake, real and regularization loss term
            l_total = hp['gan_w'] * (l_fake + l_real) + 10 * l_reg # TODO: turn into hyperparam?

            return l_total, l_fake, l_real, l_reg, acc
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

        # test at current stage: forward pass
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
        Returns:
            - xt_current: translated image at current training state of model
            - rec_current: reassembled image at current training state of model
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


    def translate_cross(self, content_image_a, content_image_b):
        """
        Params:
            - content_image_a
            - content_image_b
            - class_code_a
            - class_code_b
        Returns:
            - out: dictionary with all intermediate images and codes
        """         
        self.eval()

        xa = content_image_a.cuda()
        xb = content_image_b.cuda()          

        # forward pass
        out = self.cross_forward(xa, xb, 'test')

        return out