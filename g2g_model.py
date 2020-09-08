import copy 

import torch
import torch.nn as nn

from networks import FewShotGen, GPPatchMcResDis

def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

class G2GModel(nn.Module):

    def __init__(self, hp):
        super(G2GModel, self).__init__()
        self.gen = FewShotGen(hp['gen'])
        self.dis = GPPatchMcResDis(hp['dis'])
        self.gen_test = copy.deepcopy(self.gen)

    def cross_forward(self, xa, xb, mode):
        if mode == 'train':
            gen = self.gen 
        else:
            gen = self.gen_test

        # content and class codes image xa (meerkat)
        xa_cont = gen.enc_content(xa)
        xa_class = gen.enc_class_model(xa)

        # content and class codes image xb (dog)
        xb_cont = gen.enc_content(xb)
        xb_class = gen.enc_class_model(xb)

        # mixed images
        mb = gen.decode(xa_cont, xb_class)  # translated dog
        ma = gen.decode(xb_cont, xa_class) # translated meerkat

        # reconstruction with itself
        xa_rec = gen.decode(xa_cont, xa_class)  # reconstruction of image a (meerkat)
        xb_rec = gen.decode(xb_cont, xb_class) # reconstruction of image b (dog)
        
        # content and class codes of mixed image mb (translated dog)
        mb_cont = gen.enc_content(mb) 
        mb_class = gen.enc_class_model(mb)

        # content and class codes of mixed image ma (translated meerkat)
        ma_cont = gen.enc_content(ma)
        ma_class = gen.enc_class_model(ma)

        # reconstruction after reassembly stage
        ra = gen.decode(mb_cont, ma_class) # meerkat
        rb = gen.decode(ma_cont, mb_class) # dog

        """
        Returns (using meerkat dog example):
        Let meerkat be xa, and dog be xb, then
            - xa_cont: content code of meerkat
            - xa_class: class code of meerkat
            - xb_cont: content code of dog
            - xb_class: class code of dog
            - mb: translated dog
            - ma: translated meerkat
            - xa_rec: short reconstruction meerkat
            - xb_rec: short reconstruction dog
            - mb_cont: content code of translated dog
            - mb_class: class code of translated dog
            - ma_cont: content code of translated meerkat
            - ma_class: class code of translated meerkat
            - ra: fully reconstructed meerkat
            - rb: fully reconstructed dog
        """
        return {
            "xa_cont": xa_cont,
            "xa_class": xa_class,
            "xb_cont": xb_cont,
            "xb_class": xb_class,
            "mb": mb,
            "ma": ma,
            "xa_rec": xa_rec,
            "xb_rec": xb_rec,
            "mb_cont": mb_cont,
            "mb_class": mb_class,
            "ma_cont": ma_cont,
            "ma_class": ma_class,
            "ra": ra,
            "rb": rb,
        }

    def calc_g_loss(self, out, xa, xb, la, lb):
        # adversarial loss, generator accuracy and features
        l_adv_mb, gacc_mb, mb_gan_feat = self.dis.calc_gen_loss(out["mb"], lb) # calc_gen_loss returns loss, accuracy and gan_feat of only first param, i.e. xt
        l_adv_ma, gacc_ma, ma_gan_feat = self.dis.calc_gen_loss(out["ma"], la) # calc_gen_loss returns loss, accuracy and gan_feat of only first param, i.e. xt
        
        l_adv_xa_rec, gacc_xa_rec, xa_rec_gan_feat = self.dis.calc_gen_loss(out["xa_rec"], la)
        l_adv_xb_rec, gacc_xb_rec, xb_rec_gan_feat = self.dis.calc_gen_loss(out["xb_rec"], lb)

        # extracting features for the feature matching loss
        _, xb_gan_feat = self.dis(xb, lb) 
        _, xa_gan_feat = self.dis(xa, la) 

        # feature matching loss
        l_fm_xa_rec = recon_criterion(xa_rec_gan_feat.mean(3).mean(2), xa_gan_feat.mean(3).mean(2))
        l_fm_xb_rec = recon_criterion(xb_rec_gan_feat.mean(3).mean(2), xb_gan_feat.mean(3).mean(2)) 
        l_fm_rec = 0.5 * (l_fm_xa_rec + l_fm_xb_rec)

        l_fm_mb = recon_criterion(mb_gan_feat.mean(3).mean(2), xb_gan_feat.mean(3).mean(2))
        l_fm_ma = recon_criterion(ma_gan_feat.mean(3).mean(2), xa_gan_feat.mean(3).mean(2))
        l_fm_m = 0.5 * (l_fm_ma + l_fm_mb)

        # short reconstruction loss
        l_rec_xa = recon_criterion(out["xa_rec"], xa)
        l_rec_xb = recon_criterion(out["xb_rec"], xb)
        l_rec = 0.5 * (l_rec_xa + l_rec_xb)

        # long L1 reconstruction loss
        l_long_rec_xa = recon_criterion(out["ra"], xa)
        l_long_rec_xb = recon_criterion(out["rb"], xb)
        l_long_rec = 0.5 * (l_long_rec_xa + l_long_rec_xb)        

        # long feature matching loss
        _, gacc_long_fm_xa, ra_gan_feat = self.dis.calc_gen_loss(out["ra"], la)
        _, gacc_long_fm_xb, rb_gan_feat = self.dis.calc_gen_loss(out["rb"], lb)   
        l_long_fm_xa = recon_criterion(ra_gan_feat.mean(3).mean(2), xa_gan_feat.mean(3).mean(2))
        l_long_fm_xb = recon_criterion(rb_gan_feat.mean(3).mean(2), xb_gan_feat.mean(3).mean(2))
        l_long_fm = 0.5 * (l_long_fm_xa + l_long_fm_xb)

        # Feature matching loss in second G2G stage: between the mixed image and the reconstructed image of the same class
        l_fm_mix_rec_a = recon_criterion(ra_gan_feat.mean(3).mean(2), ma_gan_feat.mean(3).mean(2)) # compare reconstructed meerkat with mixed meerkat
        l_fm_mix_rec_b = recon_criterion(rb_gan_feat.mean(3).mean(2), mb_gan_feat.mean(3).mean(2)) # compare reconstructed dog with mixed dog
        l_fm_mix_rec = 0.5 * (l_fm_mix_rec_a + l_fm_mix_rec_b)

        # adversarial loss for 
        l_adv = 0.25 * (l_adv_ma + l_adv_mb + l_adv_xa_rec + l_adv_xb_rec)

        # accuracy
        acc = 0.25 * (gacc_ma + gacc_mb + gacc_xa_rec + gacc_xb_rec)

        # overall loss: adversarial, reconstruction and feature matching reconstruction, feature matching loss and accuracy
        return l_adv, l_rec, l_fm_rec, l_fm_m, l_long_rec, l_long_fm, l_fm_mix_rec, acc

    def calc_d_loss(self, xa, xb, la, lb, gan_weight, reg_weight):
        # calculate discriminator's real loss
        l_real_pre_a, acc_r_a, resp_r_a = self.dis.calc_dis_real_loss(xa, la)
        l_real_pre_b, acc_r_b, resp_r_b = self.dis.calc_dis_real_loss(xb, lb)
        l_real_pre = 0.5 * (l_real_pre_a + l_real_pre_b)
        l_real = gan_weight * l_real_pre
        l_real.backward(retain_graph=True)

        # real gradient penalty regularization proposed by Mescheder et al.
        l_reg_pre_a = self.dis.calc_grad2(resp_r_a, xa)
        l_reg_pre_b = self.dis.calc_grad2(resp_r_b, xb)
        l_reg_pre = 0.5 * (l_reg_pre_a + l_reg_pre_b)
        l_reg = reg_weight * l_reg_pre
        l_reg.backward()

        # generate images for the discriminator to classify
        with torch.no_grad():
            xa_cont = self.gen.enc_content(xa) # meerkat
            xb_cont = self.gen.enc_content(xb) # dog
            
            xa_class = self.gen.enc_class_model(xa)
            xb_class = self.gen.enc_class_model(xb)

            mb = self.gen.decode(xa_cont, xb_class) # dog
            ma = self.gen.decode(xb_cont, xa_class) # meerkat

        # calculate discriminator's fake loss
        l_fake_pre_a, acc_f_a, resp_f_a = self.dis.calc_dis_fake_loss(ma.detach(), lb) # meerkat
        l_fake_pre_b, acc_f_b, resp_f_b = self.dis.calc_dis_fake_loss(mb.detach(), la) # dog
        l_fake_pre = 0.5 * (l_fake_pre_a + l_fake_pre_b)
        l_fake = gan_weight * l_fake_pre
        l_fake.backward()

        acc = 0.25 * (acc_f_a + acc_f_b + acc_r_a + acc_r_b)        
        return l_fake_pre, l_real_pre, l_reg_pre, acc

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

            l_adv, l_rec, l_fm_rec, l_fm_m, l_long_rec, l_long_fm, l_fm_mix_rec, acc = self.calc_g_loss(out, xa, xb, la, lb)      

            # overall loss: adversarial, reconstruction and feature matching loss
            l_total = (hp['gan_w'] * l_adv + 
                        hp['r_w'] * l_rec + 
                        hp['fm_rec_w'] * l_fm_rec + 
                        hp['fm_w'] * l_fm_m +
                        hp['rl_w'] * l_long_rec + 
                        hp['fml_w'] * l_long_fm + 
                        hp['fml_mix_rec_w'] * l_fm_mix_rec)
            l_total.backward()

            return l_total, l_adv, l_rec, l_fm_rec, l_fm_m, l_long_rec, l_long_fm, l_fm_mix_rec, acc

        elif mode == 'dis_update':
            # for the gradient penalty regularization
            xa.requires_grad_()
            xb.requires_grad_() 

            l_fake, l_real, l_reg, acc = self.calc_d_loss(xa, xb, la, lb, hp['gan_w'], hp['reg_w'])

            # overall loss: fake, real and regularization loss term
            l_total = hp['gan_w'] * (l_fake + l_real) + l_reg

            return l_total, l_fake, l_real, l_reg, acc
        else:
            assert 0, 'Not support operation'

    def test(self, co_data, cl_data):
        """
        Params:
            - co_data: content data with one content image and one content image label
            - cl_data: class data with one class image and one class image label                    
        Returns:
            - xa: original image meerkat
            - xb: original image dog
            - mb: mixed image dog in meerkat position
            - ma: mixed image meerkat in dog position
            - ra: reconstructed image meerkat
            - rb: reconstructed image dog
        """
        self.eval()
        self.gen.eval()
        self.gen_test.eval()

        xa = co_data[0].cuda() # meerkat
        xb = cl_data[0].cuda() # dog

        out = self.cross_forward(xa, xb, 'test')

        self.train()

        return xa, xb, out['mb'], out['ma'], out['ra'], out['rb']

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

        xa_cont_current = self.gen_test.enc_content(xa)

        # perform translation for k shots
        if k == 1:
            xa_cont_current = self.gen_test.enc_content(xa)
            xb_class_current = self.gen_test.enc_class_model(xb)
            xt_current = self.gen_test.decode(xa_cont_current, xb_class_current)
        else:
            xb_class_current_before = self.gen_test.enc_class_model(xb)
            xb_class_current_after = xb_class_current_before.squeeze(-1).permute(1,
                                                                         2,
                                                                         0)
            xb_class_current_pool = torch.nn.functional.avg_pool1d(
                xb_class_current_after, k)
            xb_class_current = xb_class_current_pool.permute(2, 0, 1).unsqueeze(-1)
            xt_current = self.gen_test.decode(xa_cont_current, xb_class_current)

        return xt_current

    def compute_k_style(self, style_batch, k):      
        self.eval()
        style_batch = style_batch.cuda()
        xb_class_before = self.gen_test.enc_class_model(style_batch)
        xb_class_after = xb_class_before.squeeze(-1).permute(1, 2, 0)
        xb_class_pool = torch.nn.functional.avg_pool1d(xb_class_after, k)
        xb_class = xb_class_pool.permute(2, 0, 1).unsqueeze(-1)
        return xb_class

    def translate_simple(self, content_image, class_code):
        self.eval()
        xa = content_image.cuda()
        xb_class_current = class_code.cuda()
        xa_cont_current = self.gen_test.enc_content(xa)
        xt_current = self.gen_test.decode(xa_cont_current, xb_class_current)
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
