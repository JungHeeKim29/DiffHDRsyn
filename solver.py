import time
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch.utils.data

from PIL import Image

from torch import nn
from torch import optim

from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from dataloader import ImageDataset
from cobi_loss import ContextualBilateralLoss as cx

from config_tool import get_config

from hdrlayer import hdrlayer
from model.GRU_model import Generator_up, Generator_down
from model.structure import Structure_up, Structure_down, CombineNet_up, CombineNet_down

from utils.io import *
from utils.Adam import Adam_GCC
from utils.img_utils import *

class Solver(object):

    def __init__(self, config):
        super(Solver, self).__init__()
        # Training options
        self.batch_size = config['batch_size'] # 32
        self.img_size = config['img_size']
        self.n_epochs = config['n_epochs']

        self.min_ev = config['min_ev'] # -3
        self.max_ev = config['max_ev'] #  3

        self.ref_ev = (self.max_ev-self.min_ev)//2
        self.length = self.max_ev - self.min_ev + 1

        self.n_epochs = config['n_epochs']
        self.lr_edge = config['lr_edge']
        self.lr_glob = config['lr_glob']
        self.lr_rend = config['lr_render']
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
        self.optim = config['optim']        
        self.decay = config['decay']
        self.data_dir = config['data_dir']
        self.validate_dir = config['validate_dir']
  
        self.sample_dir = config['sample_dir']
        self.model_path = config['model_dir']
        
        self.pretrained = config['pretrained']
        # Configurations
        self.device = config['gpu_id']
        self.dataloader = DataLoader(ImageDataset(
                                     self.data_dir, self.img_size,
                                     min_ev = int(self.min_ev), 
                                     max_ev = self.max_ev,
                                     fliplr = True, fliptb = False,
                                     rot = False, randcrop = True,
                                     transform=True, 
                                     mode = 'train'), 
                                     batch_size = self.batch_size, 
                                     shuffle=True)

        self.validloader = DataLoader(ImageDataset(
                                     self.data_dir, self.img_size,
                                     min_ev = int(self.min_ev),
                                     max_ev = self.max_ev,
                                     fliplr = False, fliptb = False,
                                     rot = False, randcrop = False,
                                     transform=True,
                                     mode = 'test'),
                                     batch_size = 1,
                                     shuffle=False)

        # Steps to log, validate, sample
        self.log_step = config['log_step']
        self.validate_step = config['validate_step']
        self.model_save_step = config['model_save_step']
        self.load_step = config['load_step']
        self.display_step = config['display_step']
        self.refine_step = config['refine_step']

        self.recon_loss = nn.L1Loss()      
        self.hdr_loss = nn.MSELoss()
        self.cx_loss1 = cx(use_vgg = True, vgg_layer = 'relu3_4', 
                           device = self.device[1])
        self.cx_loss2 = cx(use_vgg = True, vgg_layer = 'relu4_4',
                           device = self.device[1])

        self.build_model()
 
    def build_model(self):
        self.gen_up = Generator_up(3,32,5,1, self.length-1).cuda(device = self.device[0])
        self.gen_down = Generator_down(3,32,5,1, self.length-1).cuda(device = self.device[0])

        self.struct_up = Structure_up(2,16,4,1, self.length-1)\
                            .cuda(device = self.device[1])
        self.struct_down = Structure_down(2,16,4,1, self.length-1)\
                            .cuda(device = self.device[1])

        self.combine_up = CombineNet_up(7,16,5,1)\
                            .cuda(device = self.device[1])
        self.combine_down = CombineNet_down(7,16,5,1)\
                            .cuda(device = self.device[1])
        self.hdrlayer_pred = hdrlayer(lin_type = 'LUT', method='Debevec',device = self.device[1])          
        self.hdrlayer_gt = hdrlayer(lin_type = 'LUT', method='Debevec',device = self.device[1])

        if self.pretrained : 
            print('Using pre-trained weights')
            gen_up_path = os.path.join(self.model_path,
                         '{}-gu.ckpt'.format(self.load_step))
            gen_down_path = os.path.join(self.model_path,
                         '{}-gd.ckpt'.format(self.load_step))
            struct_up_path = os.path.join(self.model_path,
                         '{}-su.ckpt'.format(self.load_step))
            struct_down_path = os.path.join(self.model_path,
                         '{}-sd.ckpt'.format(self.load_step))
            combine_up_path = os.path.join(self.model_path,
                         '{}-cbu.ckpt'.format(self.load_step))
            combine_down_path = os.path.join(self.model_path,
                         '{}-cbd.ckpt'.format(self.load_step))
   
            self.gen_up.load_state_dict(torch.load(gen_up_path))
            self.gen_down.load_state_dict(torch.load(gen_down_path))
            self.struct_up.load_state_dict(torch.load(struct_up_path))
            self.struct_down.load_state_dict(torch.load(struct_down_path))
            self.combine_up.load_state_dict(torch.load(combine_up_path))
            self.combine_down.load_state_dict(torch.load(combine_down_path))

        if self.optim == 'adam':

            self.optim_gu = optim.Adam(self.gen_up.parameters(),
                                      lr = self.lr_glob,
                                      betas=(self.beta1, self.beta2),
                                      weight_decay = self.decay)
            self.optim_gd = optim.Adam(self.gen_down.parameters(),
                                      lr = self.lr_glob,
                                      betas=(self.beta1, self.beta2),
                                      weight_decay = self.decay)
            self.optim_su = optim.Adam(self.struct_up.parameters(),
                                      lr = self.lr_edge,
                                      betas=(self.beta1, self.beta2),
                                      weight_decay = self.decay)
            self.optim_sd = optim.Adam(self.struct_down.parameters(),
                                      lr = self.lr_edge,
                                      betas=(self.beta1, self.beta2),
                                      weight_decay = self.decay)
            self.optim_cbu = optim.Adam(self.combine_up.parameters(),
                                      lr = self.lr_rend,
                                      betas=(self.beta1, self.beta2),
                                      weight_decay = self.decay)
            self.optim_cbd = optim.Adam(self.combine_down.parameters(),
                                      lr = self.lr_rend,
                                      betas=(self.beta1, self.beta2),
                                      weight_decay = self.decay)

        elif self.optim == 'adam_cent':
            self.optim_gu = Adam_GCC(self.gen_up.parameters(),
                                      lr = self.lr_glob,
                                      betas=(self.beta1, self.beta2),
                                      weight_decay = self.decay)
            self.optim_gd = Adam_GCC(self.gen_down.parameters(),
                                      lr = self.lr_glob,
                                      betas=(self.beta1, self.beta2),
                                      weight_decay = self.decay) 
                 
            self.optim_su = Adam_GCC(self.struct_up.parameters(),
                                      lr = self.lr_edge,
                                      betas=(self.beta1, self.beta2),
                                      weight_decay = self.decay)
            self.optim_sd = Adam_GCC(self.struct_down.parameters(),
                                      lr = self.lr_edge,
                                      betas=(self.beta1, self.beta2),
                                      weight_decay = self.decay)
            self.optim_cbu = Adam_GCC(self.combine_up.parameters(),
                                      lr = self.lr_rend,
                                      betas=(self.beta1, self.beta2),
                                      weight_decay = self.decay)
            self.optim_cbd = Adam_GCC(self.combine_down.parameters(),
                                      lr = self.lr_rend,
                                      betas=(self.beta1, self.beta2),
                                      weight_decay = self.decay)       

    def print_network(self, model, name) :
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def update_lr(self, edge_lr, int_lr, g_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.optim_gu.param_groups:
            param_group['lr'] = int_lr
        for param_group in self.optim_gd.param_groups:
            param_group['lr'] = int_lr
        for param_group in self.optim_su.param_groups:
            param_group['lr'] = edge_lr
        for param_group in self.optim_sd.param_groups:
            param_group['lr'] = edge_lr
        for param_group in self.optim_cbu.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.optim_cbd.param_groups:
            param_group['lr'] = g_lr

    def reset_grad(self):
        # Reset all gradients
        self.optim_gu.zero_grad()
        self.optim_gd.zero_grad()
        self.optim_su.zero_grad()
        self.optim_sd.zero_grad()
        self.optim_cbu.zero_grad()
        self.optim_cbd.zero_grad()

    def Up_ev_model(self, img, edge, style_id = 0, h_img = None, h_edge = None):
        glob_up_img, h_img = self.gen_up(img.to(self.device[0]), 
                                         style_id,
                                         h_img)

        img_gray = torch.mean(img, dim=1).unsqueeze(1)\
                   .to(self.device[1])
        up_edge, h_edge  = self.struct_up(torch.cat((img_gray, edge), 
                                        dim=1).to(self.device[1]),
                                        style_id,
                                        h_edge)
        
        concat_images = torch.cat((img.to(self.device[1]),
                                   glob_up_img.to(self.device[1]), 
                                   up_edge),
                                   dim=1)
        up_img = self.combine_up(concat_images)

        return up_img, glob_up_img, up_edge, h_img, h_edge

    def Down_ev_model(self, img, edge, style_id = 0, h_img=None, h_edge = None):
        glob_down_img, h_img = self.gen_down(img.to(self.device[0]), 
                                             style_id,
                                             h_img)

        img_gray = torch.mean(img, dim=1).unsqueeze(1)\
                   .to(self.device[1])

        down_edge, h_edge = self.struct_down(torch.cat(
                            (img_gray, edge),
                            dim=1).to(self.device[1]), 
                            style_id,
                            h_edge)

        concat_images = torch.cat((img.to(self.device[1]),
                                   glob_down_img.to(self.device[1]), 
                                   down_edge),
                                   dim=1)
        down_img = self.combine_down(concat_images)

        return down_img, glob_down_img, down_edge, h_img, h_edge

    def HDR_model(self, img_stack, edge_stack, ref_ev, 
                  up_length, down_length, step):

        pred_stack = torch.zeros_like(img_stack)
        pred_int_stack = torch.zeros_like(img_stack)
        pred_edge = torch.zeros_like(edge_stack)

        up_img = img_stack[:, ref_ev, :]
        up_int_img = img_stack[:, ref_ev, :]
        down_img = img_stack[:, ref_ev, :]
        down_int_img = img_stack[:, ref_ev, :]
        up_edge = edge_stack[:, ref_ev, :]
        down_edge = edge_stack[:, ref_ev,:]
       

        pred_stack[:, ref_ev,:] = img_stack[:, ref_ev, :]
        pred_int_stack[:, ref_ev,:] = img_stack[:, ref_ev,:]
        pred_edge[:, ref_ev, :] = edge_stack[:,ref_ev,:]
        
        up_h_img = None
        down_h_img = None
        up_h_edge = None
        down_h_edge = None

        for up in range(up_length):
            up_index = ref_ev + up + 1

            if step <= self.refine_step :
                up_img, up_int_img, up_edge, up_h_img, up_h_edge \
                = self.Up_ev_model(up_int_img, up_edge,up_index-1, up_h_img, up_h_edge)
            else:
                up_img, up_int_img, up_edge, up_h_img, up_h_edge \
                = self.Up_ev_model(up_img, up_edge,up_index-1, up_h_img, up_h_edge)

            pred_stack[:, up_index, :] = up_img
            pred_int_stack[:, up_index, :] = up_int_img
            pred_edge[:, up_index,:] = up_edge

        for down in range(down_length):
            down_index = ref_ev - down -1
            if step <= self.refine_step:
                down_img, down_int_img, down_edge, down_h_img, down_h_edge\
                = self.Down_ev_model(down_int_img, down_edge, 
                                     down_index,down_h_img, down_h_edge)
            else:
                down_img, down_int_img, down_edge, down_h_img, down_h_edge\
                = self.Down_ev_model(down_img, down_edge,
                                     down_index,down_h_img, down_h_edge)

            pred_stack[:, down_index, :] = down_img
            pred_int_stack[:, down_index, :] = down_int_img
            pred_edge[:,down_index, :] = down_edge       
    
        return pred_stack, pred_int_stack, pred_edge

    def validate(self, step):
        sample_path = os.path.join(self.sample_dir,'val_'+str(step))
        if not (os.path.isdir(sample_path)):
            os.mkdir(sample_path)

        validate_imgs = list()

        tot_tonemap_psnr = 0
        
        pred_psnr = torch.zeros(self.length-1)
        
        dtype = torch.cuda.FloatTensor
		
        # Validation
        for index, data in enumerate(self.validloader):

            scene = 't'+str(index)
            scene_path = os.path.join(sample_path, str(scene))

            if not (os.path.isdir(scene_path)):
                os.mkdir(scene_path)

            # Inputs
            image_stack = data[0].type(dtype)
            target_hdr = data[1].type(dtype)
            ev_stack = data[2].type(dtype)
            edge_stack = data[3].type(dtype)

            image_stack = image_stack.to(self.device[0])
            target_hdr = target_hdr.to(self.device[1])
            ev_stack = ev_stack.to(self.device[1])
            edge_stack = edge_stack.to(self.device[1])
            
            pred_stack, pred_int_stack, pred_edge_stack\
            = self.HDR_model(image_stack, edge_stack, self.ref_ev,
                                  up_length = 3, down_length=3, 
                                  step=step)

            if step <= self.refine_step : 
                pred_stack = pred_int_stack

            psnr = torch.tensor([compute_psnr(image_stack[:,i,:].cpu(),
                    pred_stack[:,i,:].cpu(), 1.0) \
                    for i in [0,1,2,4,5,6]]) 
            pred_psnr += psnr
            denorm_pred_stack = [inv_transform(pred_stack[:,i,:].cpu())\
                                 for i in [0,1,2,4,5,6]]
            denorm_gt_stack = [inv_transform(image_stack[:,i,:].cpu())\
                                 for i in [0,1,2,4,5,6]]
            index = 0
            for i in range(6):
                save_image = np.concatenate((denorm_pred_stack[i],
		                                      denorm_gt_stack[i]),
                                                  axis = 1)
                save_image = Image.fromarray(save_image)

                if i == 3:
                    index +=  1

                output_up_name = os.path.join(scene_path,
                        str(index - self.ref_ev)+'EV.png')
                              
                save_image.save(output_up_name)
                index += 1

            gen_hdr_stack,_ = self.hdrlayer_pred(\
                           ((pred_stack+1)/2).to(self.device[1]),
                            ev_stack.to(self.device[1]))
            target_hdr,_ = self.hdrlayer_gt(\
                           ((image_stack+1)/2).to(self.device[1]), 
                            ev_stack.to(self.device[1]))
                
            pred_tone_map = tone_map(gen_hdr_stack.cpu())
            gt_tone_map = tone_map(target_hdr.cpu())

            pred_tone_map = cv2.cvtColor(pred_tone_map, 
                                         cv2.COLOR_BGR2RGB)
            gt_tone_map = cv2.cvtColor(gt_tone_map, cv2.COLOR_BGR2RGB)

            tone_map_psnr = tone_psnr(pred_tone_map, gt_tone_map, 
			                           pixel_range=255)
            pred_tone_map = np.concatenate((pred_tone_map, gt_tone_map),
                                            axis = 1)

            tot_tonemap_psnr += tone_map_psnr
            tone_map_name = os.path.join(scene_path, 
                                        'pred_tone_map.png')
            cv2.imwrite(tone_map_name, pred_tone_map)
              
        pred_psnr /= len(self.validloader)
        tot_tonemap_psnr /= len(self.validloader)
        print('Validate tone_map PSNR : {}'.format(tot_tonemap_psnr))
        print('Validate up +1 generator PSNR : {}'.format(pred_psnr[3]))
        print('Validate up +2 generator PSNR : {}'.format(pred_psnr[4]))
        print('Validate up +3 generator PSNR : {}'.format(pred_psnr[5]))
        print('Validate down -1 generator PSNR : {}'.format(pred_psnr[2]))
        print('Validate down -2  generator PSNR : {}'.format(pred_psnr[1]))
        print('Validate down -3 generator PSNR : {}'.format(pred_psnr[0]))
        
    def train(self):
        
        dataloader = self.dataloader       
        dtype = torch.cuda.FloatTensor

        non_zero_offset = 1e-6
        start_time = time.time()

        if self.pretrained:
            iter_num = self.load_step

        else:

            iter_num = 0

        for epoch in range(self.n_epochs):
            for step, data in enumerate(dataloader):

                # Preprocessing
                # Inputs
                image_stack = data[0].type(dtype)
                target_hdr = data[1].type(dtype)
                ev_stack = data[2].type(dtype)
                edge_stack = data[3].type(dtype)
 
                image_stack = image_stack.to(self.device[1])
                target_hdr = target_hdr.to(self.device[1])
                ev_stack = ev_stack.to(self.device[1])
                edge_stack = edge_stack.to(self.device[1])                 

                loss = {}

                single_gu_loss = 0
                single_eu_loss = 0
                single_gd_loss = 0
                single_ed_loss = 0
                single_cu_loss = 0
                single_cd_loss = 0
                single_iu_loss = 0
                single_id_loss = 0
 
                pred_int_stack = torch.zeros_like(image_stack)
                pred_stack = torch.zeros_like(image_stack)
                pred_edge_stack = torch.zeros_like(edge_stack)
 
                ev = random.randrange(-3,4) + self.ref_ev

                up_length = 1e-8
                down_length = 1e-8

                ref_ev_up = ev#self.ref_ev
                ref_ev_down = ev#self.ref_ev

                ref_up_image = image_stack[:,ref_ev_up,:]
                ref_down_image = image_stack[:, ref_ev_down,:]

                ref_up_edge = edge_stack[:,ref_ev_up,:]
                ref_down_edge = edge_stack[:,ref_ev_down,:]

                h_up_img = None
                h_up_edge = None
                h_down_img = None
                h_down_edge = None


                for index in range((self.length-1)) : 
                    # Generator Up
                    up_index = index + 1 + ev

                    if up_index <= self.length - 1:

                        image_up_t = image_stack[:,up_index,:]
                        edge_up_t = edge_stack[:,up_index, :]
                        image_genup, inter_genup, edge_genup,\
                        h_up_img, h_up_edge\
                        =  self.Up_ev_model(ref_up_image, 
                                            ref_up_edge, 
                                            up_index-1,
                                            h_up_img,
                                            h_up_edge)

                        h_up_img = h_up_img.detach()
                        h_up_edge = h_up_edge.detach()

                        gen_img_hist = extract_hist(\
                                       inter_genup.to(self.device[1]))
                        t_img_hist = extract_hist(image_up_t)
                        r_hist_loss = self.recon_loss(gen_img_hist,
                                      t_img_hist.to(self.device[1]))

                        r_int_img_loss \
                        = self.recon_loss(inter_genup.to(
                                          self.device[1]),
                                          image_up_t)

                        r_edge_loss = self.recon_loss(edge_genup, 
                                                      edge_up_t)
                        single_up_int_loss = r_int_img_loss \
                                             + 0.01*r_hist_loss

                        single_iu_loss += single_up_int_loss.item()
                        single_eu_loss += r_edge_loss.item()
       
                        if iter_num > self.refine_step:
                            r_img_loss = self.recon_loss(image_genup,
                                                 image_up_t)

                            cx_loss = self.cx_loss1((image_genup+1)/2,
                                               (image_up_t+1)/2)
                            cx_loss += self.cx_loss2((image_genup+1)/2,
                                                (image_up_t+1)/2)

                            single_up_loss= r_img_loss + 0.1*cx_loss
                             
                            self.reset_grad()
                            r_edge_loss.backward(retain_graph = True)
                            self.optim_su.step()

                            self.reset_grad()
                            single_up_int_loss.backward(retain_graph = True)
                            self.optim_gu.step()
   
                            self.reset_grad()
                            single_up_loss.backward()
                            self.optim_cbu.step()

                            single_gu_loss += single_up_loss.item()

                        else:
                            self.reset_grad()
                            r_edge_loss.backward(retain_graph = True)
                            self.optim_su.step()

                            self.reset_grad()
                            single_up_int_loss.backward()
                            self.optim_gu.step()
 
                        ref_up_image = image_up_t
                        ref_up_edge = edge_up_t                   

                        up_length += 1

                    # Generator Down
                    down_index = ev - index -1 
                    if down_index >= 0 :
                        image_down_t = image_stack[:,down_index,:]
                        edge_down_t = edge_stack[:,down_index,:]

                        image_gendown, inter_gendown, edge_gendown,\
                        h_down_img, h_down_edge\
                         = self.Down_ev_model(ref_down_image,
                                             ref_down_edge,
                                             down_index,
                                             h_down_img,
                                             h_down_edge)
                    
                        h_down_img = h_down_img.detach()
                        h_down_edge = h_down_edge.detach()
 
                        gen_img_hist = extract_hist(inter_gendown)
                        t_img_hist = extract_hist(image_down_t)

                        r_hist_loss = self.recon_loss(\
                                      gen_img_hist.to(self.device[1]),
                                      t_img_hist)

                        r_int_img_loss \
                        = self.recon_loss(inter_gendown.to(
                                           self.device[1]), 
                                           image_down_t)

                        r_edge_loss = self.recon_loss(edge_gendown, 
                                                     edge_down_t)

                        single_down_int_loss = r_int_img_loss \
                                               + 0.01*r_hist_loss

                        single_id_loss += single_down_int_loss.item()       
                        single_ed_loss += r_edge_loss.item()

                        if iter_num > self.refine_step:
                            r_img_loss = self.recon_loss(image_gendown,
                                                         image_down_t)

                            cx_loss = self.cx_loss1((image_gendown+1)/2,
                                                    (image_down_t+1)/2)
                            cx_loss += self.cx_loss2((image_gendown+1)/2,
                                                    (image_down_t+1)/2)

                            single_down_loss= r_img_loss +0.1*cx_loss

                            self.reset_grad()
                            r_edge_loss.backward(retain_graph = True)
                            self.optim_sd.step()
 
                            self.reset_grad()
                            single_down_int_loss.backward(retain_graph = True)
                            self.optim_gd.step()

                            self.reset_grad()
                            single_down_loss.backward()
                            self.optim_cbd.step()

                            single_gd_loss += single_down_loss.item()
  
                        else :
                            self.reset_grad()
                            r_edge_loss.backward(retain_graph = True)
                            self.optim_sd.step()

                            self.reset_grad()
                            single_down_int_loss.backward()
                            self.optim_gd.step()

                        ref_down_image = image_down_t
                        ref_down_edge = edge_down_t
                      
                        down_length += 1

                if iter_num > self.refine_step:
                    loss['GU/single_loss'] = single_gu_loss/up_length
                    loss['GD/single_loss'] = single_gd_loss/down_length

                loss['GU/single_int_loss'] = single_iu_loss/up_length
                loss['GU/edge_loss'] = single_eu_loss/up_length
                loss['GD/single_int_loss'] = single_id_loss/down_length
                loss['GD/edge_loss'] = single_ed_loss/down_length

                total_stack_loss = 0
                total_int_stack_loss = 0
                total_edge_loss = 0
                total_hdr_loss = 0 

                image_stack = image_stack.to(self.device[1])

                # Generate stack from centered ev
                ev = random.randrange(-1,2)
                ref_ev = self.ref_ev + ev
                        
                pred_stack, pred_int_stack, pred_edge_stack \
                = self.HDR_model(image_stack, edge_stack, 
                                     ref_ev,
                                     up_length = self.max_ev-ev, 
                                     down_length = ev - self.min_ev, 
                                     step = iter_num)

                gen_hist = extract_hist(pred_int_stack)
                target_hist = extract_hist(image_stack)
                     
                stack_hist_loss = self.recon_loss(\
                                         gen_hist.to(self.device[1]),
                                         target_hist)
                        
                stack_int_recon_loss = self.recon_loss(\
                              pred_int_stack.to(self.device[1]),
                              image_stack)
                                              
                edge_recon_loss = self.recon_loss(\
                              pred_edge_stack, 
                              edge_stack)
                       
                stack_int_loss = stack_int_recon_loss\
                                         + 0.01*stack_hist_loss 
                
                total_int_stack_loss += stack_int_loss.item()
                total_edge_loss += edge_recon_loss.item()
                    
                if iter_num > self.refine_step:

                    gen_hdr, gen_crf = self.hdrlayer_pred(\
                             ((pred_stack+1)/2).to(self.device[1]),
                               ev_stack)
                    #target_hdr, target_crf = self.hdrlayer_gt(\
                    #         ((image_stack+1)/2).to(self.device[1]),
                    #           ev_stack)

                    stack_recon_loss=self.recon_loss(\
                                         pred_stack, image_stack)

                    stack_cx_loss = torch.tensor([self.cx_loss1(\
                                         (pred_stack[:,i,:]+1)/2,
                                         (image_stack[:,i,:]+1)/2)
                                         for i in range(self.length)]).mean()

                    stack_cx_loss += torch.tensor(\
                                         [self.cx_loss2(\
                                         (pred_stack[:,i,:]+1)/2,
                                         (image_stack[:,i,:]+1)/2)
                                          for i in range(self.length)]).mean()

                    hdr_loss = self.recon_loss(mu_law(gen_hdr, mu = 100),
                                                mu_law(target_hdr, mu = 100))
 
                    stack_loss = stack_recon_loss + 0.1*stack_cx_loss

                    total_stack_loss += stack_loss.item()
                    total_hdr_loss += hdr_loss.item()
    
                    self.reset_grad()
                    edge_recon_loss.backward(retain_graph=True)
                    self.optim_su.step()
                    self.optim_sd.step()

                    self.reset_grad()
                    stack_int_loss.backward(retain_graph=True)
                    self.optim_gu.step()
                    self.optim_gd.step()

                    self.reset_grad() 
                    stack_loss.backward(retain_graph = True)
                    self.optim_cbu.step()
                    self.optim_cbd.step()

                    if iter_num > self.refine_step: 
                        self.reset_grad()
                        hdr_loss.backward()
                        self.optim_cbu.step()
                        self.optim_cbd.step()
                        self.optim_gu.step()
                        self.optim_gd.step()
                        self.optim_su.step()
                        self.optim_sd.step()

                    loss['HDR/LogHDR_loss'] \
                    = total_stack_loss
                    loss['HDR/HDR_loss'] \
                    = total_hdr_loss
                    loss['HDR/CXloss'] \
                    = stack_cx_loss.item()
                else:
                    self.reset_grad()
                    edge_recon_loss.backward(retain_graph=True)
                    self.optim_su.step()
                    self.optim_sd.step()

                    self.reset_grad()
                    stack_int_loss.backward()
                    self.optim_gu.step()
                    self.optim_gd.step()

                loss['HDR/HDR_int_loss'] \
                = total_int_stack_loss
                loss['HDR/edge_loss'] \
                = total_edge_loss 

                '''
                   Print log
                '''

                # Print out training information.
                if (iter_num+1) % self.log_step == 0:
                    et = time.time() - start_time
                    log = "Elapsed [{}], Iteration [{}]".format(
                                             et, iter_num+1)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)


                if (iter_num+1) % self.validate_step ==0:
                    with torch.no_grad():
                        self.validate(iter_num)   

                # Save model checkpoints.
                if (iter_num+1) % self.model_save_step == 0:
                    gu_path = os.path.join(self.model_path, 
                                      '{}-gu.ckpt'.format(iter_num+1))
                    gd_path = os.path.join(self.model_path,
                                      '{}-gd.ckpt'.format(iter_num+1))
                    su_path = os.path.join(self.model_path,
                                      '{}-su.ckpt'.format(iter_num+1))
                    sd_path = os.path.join(self.model_path,
                                      '{}-sd.ckpt'.format(iter_num+1))

                    cbu_path = os.path.join(self.model_path,
                                      '{}-cbu.ckpt'.format(iter_num+1))
                    cbd_path = os.path.join(self.model_path,
                                      '{}-cbd.ckpt'.format(iter_num+1))

                    torch.save(self.gen_up.state_dict(), gu_path)
                    torch.save(self.gen_down.state_dict(), gd_path)
                    torch.save(self.struct_up.state_dict(), su_path)
                    torch.save(self.struct_down.state_dict(), sd_path)
                    torch.save(self.combine_up.state_dict(), cbu_path)
                    torch.save(self.combine_down.state_dict(), cbd_path)

                    print('Saved model checkpoints into {}...'.format(
                                                   self.model_path))

                # Display interim results
                if (iter_num+1) % self.display_step == 0:
                    image = torch.cat([image_stack.squeeze(0),
                                       pred_int_stack.squeeze(0),
                                       pred_stack.squeeze(0)],0)
                    edge = torch.cat([(edge_stack*2-1).squeeze(0),
                                      (pred_edge_stack*2-1).squeeze(0)], 0)
                    edge = edge.expand([self.length*2, 3, 
                           self.img_size, self.img_size])
                    
                    image_edge = torch.cat([image, edge], 0)

                    psnr_int_up = compute_psnr(image_stack[:, self.ref_ev+1:,:],
                                             pred_int_stack[:,self.ref_ev+1:,:])
                    psnr_int_down = compute_psnr(image_stack[:, :self.ref_ev, :],
                                             pred_int_stack[:,:self.ref_ev,:])

                    image_edge_grid = make_grid(image_edge,
                                          nrow = self.length)

                    if iter_num > self.refine_step  :

                        psnr_up = compute_psnr(image_stack[:,self.ref_ev+1:,:],
                                                pred_stack[:,self.ref_ev+1:,:])

                        psnr_down = compute_psnr(image_stack[:, :self.ref_ev,:],
                                                 pred_stack[:,:self.ref_ev, :])

                        hdr_t = tone_map(target_hdr.detach().cpu())
                        hdr_g = tone_map(gen_hdr.detach().cpu())

                        psnr_tonemap = tone_psnr(hdr_t, hdr_g)

                        hdr = torch.cat([torch.from_numpy(hdr_t), 
						                torch.from_numpy(hdr_g)], 1)
                        image_grid2 = make_grid(hdr,
                                           nrow = self.batch_size)


                        print('PSNR of tonemapped image : {}'\
                               .format(psnr_tonemap))
                        print('PSNR of up sample image : {}'\
                               .format(psnr_up))
                        print('PSNR of down sample image : {}'\
                               .format(psnr_down))
                        print('PSNR of int_up sample image : {}'\
                               .format(psnr_int_up))
                        print('PSNR of int_down sample image : {}'\
                               .format(psnr_int_down))

                    else:

                        image_grid2 = image_edge_grid 
                        image_grid2 = (image_grid2+1)/2
                        image_grid2 = image_grid2.detach().cpu()
                        image_grid2 = image_grid2.numpy()
                        image_grid2 = np.transpose(image_grid2,(1,2,0))
                           
                        print('PSNR of int_up sample image : {}'\
                              .format(psnr_int_up))
                        print('PSNR of int_down sample image : {}'\
                              .format(psnr_int_down))

              
                    matplotlib_imshow(image_edge_grid, image_grid2,
                                           one_channel=False) 

                iter_num +=1

