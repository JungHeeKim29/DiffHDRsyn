import time
import cv2
import os
import random
import numpy as np

import torch.utils.data
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from dataloader import ImageDataset
from config_tool import get_config

from hdrlayer import hdrlayer
from model.GRU_model import Generator_up, Generator_down
from model.structure import Structure_up, Structure_down
from model.combine import CombineNet_up, CombineNet_down

from utils.io import *
from utils.img_utils import *

class test_HDR():
    def __init__(self, config):
        super(test_HDR, self).__init__()
        self.batch_size = config['batch_size'] # 32
        self.img_size = config['img_size']
        self.min_ev = config['min_ev'] # -3
        self.max_ev = config['max_ev'] #  3
        self.ref_ev = config['ref_ev'] 
        self.length = self.max_ev - self.min_ev + 1

        self.data_dir = config['test_dir']

        self.device = config['gpu_id']
        self.model_path = config['model_dir']
        self.output_dir = config['test_result_dir']

        self.test_loader = DataLoader(ImageDataset(
                                     self.data_dir, self.img_size,
                                     min_ev = int(self.min_ev),
                                     max_ev = self.max_ev,
                                     fliplr = False, fliptb = False,
                                     rot = False, randcrop = False,
                                     transform=True,
                                     mode = 'test'),
                                     batch_size = 1,
                                     shuffle=False)

        self.load_step = config['test_step']
        self.build_model()

    def build_model(self):
        self.gen_up = Generator_up(3,32,5,1, 7-1)
        self.gen_down = Generator_down(3,32,5,1, 7-1)

        self.struct_up = Structure_up(2,16,4,1, 7-1)
        self.struct_down = Structure_down(2,16,4,1, 7-1)

        self.combine_up = CombineNet_up(7,16,5,1)
        self.combine_down = CombineNet_down(7,16,5,1)
        self.hdrlayer = hdrlayer(lin_type = 'LUT', method='Debevec')

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

    def Up_ev_model(self, img, edge, style_id = 0, h_img = None, h_edge = None):
        glob_up_img, h_img = self.gen_up(img,
                                         style_id,
                                         h_img)

        img_gray = torch.mean(img, dim=1).unsqueeze(1)
        up_edge, h_edge  = self.struct_up(torch.cat((img_gray, edge),
                                        dim=1),
                                        style_id,
                                        h_edge)

        concat_images = torch.cat((img, glob_up_img,
                                   up_edge),
                                   dim=1)
        up_img = self.combine_up(concat_images)

        return up_img, glob_up_img, up_edge, h_img, h_edge

    def Down_ev_model(self, img, edge, style_id = 0, h_img=None, h_edge = None):
        glob_down_img, h_img = self.gen_down(img,
                                             style_id,
                                             h_img)

        img_gray = torch.mean(img, dim=1).unsqueeze(1)

        down_edge, h_edge = self.struct_down(torch.cat(
                            (img_gray, edge),
                            dim=1),
                            style_id,
                            h_edge)

        concat_images = torch.cat((img,
                                   glob_down_img,
                                   down_edge),
                                   dim=1)
        down_img = self.combine_down(concat_images)

        return down_img, glob_down_img, down_edge, h_img, h_edge

    def HDR_model(self, img_stack, edge_stack, ref_ev,
                  up_length, down_length):

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
            
            up_img, up_int_img, up_edge, up_h_img, up_h_edge \
            = self.Up_ev_model(up_img, up_edge,up_index-1, up_h_img, up_h_edge)

            pred_stack[:, up_index, :] = up_img
            pred_int_stack[:, up_index, :] = up_int_img
            pred_edge[:, up_index,:] = up_edge

        for down in range(down_length):
            down_index = ref_ev - down -1
            down_img, down_int_img, down_edge, down_h_img, down_h_edge\
            = self.Down_ev_model(down_img, down_edge,
                                 down_index,down_h_img, down_h_edge)

            pred_stack[:, down_index, :] = down_img
            pred_int_stack[:, down_index, :] = down_int_img
            pred_edge[:,down_index, :] = down_edge

        return pred_stack, pred_int_stack, pred_edge

    def test(self):

        sample_path = os.path.join(self.output_dir,'test')        
        if not (os.path.isdir(sample_path)):
            os.mkdir(sample_path)

        validate_imgs = list()

        dtype = torch.FloatTensor

        # Validation
        for index, data in enumerate(self.test_loader):

            scene = '%04d' % index
            scene_path = os.path.join(sample_path, str(scene))

            if not (os.path.isdir(scene_path)):
                os.mkdir(scene_path)

            # Inputs
            image_stack = data[0].type(dtype)
            #target_hdr = data[1].type(dtype)
            ev_stack = data[2].type(dtype)
            edge_stack = data[3].type(dtype)
           
            up_length = (self.max_ev-self.min_ev) - self.ref_ev 
            down_length = self.ref_ev
            pred_stack, pred_int_stack, pred_edge_stack\
            = self.HDR_model(image_stack, edge_stack, self.ref_ev,
                                  up_length = up_length, down_length=down_length)

            denorm_pred_stack = [inv_transform(pred_stack[:,i,:].cpu())\
                                 for i in [0,1,2,3,4,5,6]]
            denorm_image_stack = [inv_transform(image_stack[:,i,:].cpu())\
                                 for i in [0,1,2,3,4,5,6]]
            denorm_pred_edge = [inv_transform(\
                                ((pred_edge_stack[:,i,:]-0.5)*2).cpu())\
                                for i in [0,1,2,3,4,5,6]]
            denorm_target_edge = [inv_transform(\
                                ((edge_stack[:,i,:]-0.5)*2).cpu())\
                                for i in [0,1,2,3,4,5,6]]
            denorm_pred_int_stack = [inv_transform(pred_int_stack[:,i,:].cpu())\
                                for i in [0,1,2,3,4,5,6]]
              

            index = 0
            for i in range(7):
                pred_image = denorm_pred_stack[i]
                target_image = denorm_image_stack[i]

                pred_edge = denorm_pred_edge[i]
                target_edge = denorm_target_edge[i]

                pred_output_name = os.path.join(scene_path,
                     str(index - self.ref_ev)+'EV_pred.jpg.png')
                pred_edge_name = os.path.join(scene_path,
                     str(index - self.ref_ev)+'EV_pred_edge.png')

                pred_image.save(pred_output_name)
                pred_edge.save(pred_edge_name)

                index += 1

            gen_hdr_stack,gen_crf = self.hdrlayer(\
                           ((pred_stack+1)/2),
                            ev_stack)
            target_hdr,target_crf = self.hdrlayer(\
                           ((image_stack+1)/2),
                            ev_stack)

            gen_hdr_stack = gen_hdr_stack.detach()
            target_hdr = target_hdr.detach()
            denorm_gen_hdr = np.array(gen_hdr_stack\
                                      .squeeze().permute(1,2,0).cpu())
            denorm_target_hdr = np.array(target_hdr\
                                      .squeeze().permute(1,2,0).cpu())
             
            denorm_gen_hdr = denorm_gen_hdr[:,:,::-1]
            denorm_target_hdr = denorm_target_hdr[:,:,::-1]

            gt_tone_map = tone_map(target_hdr.cpu())
            pred_tone_map = tone_map(gen_hdr_stack.cpu())

            gt_tone_map = cv2.cvtColor(gt_tone_map, 
                                       cv2.COLOR_BGR2RGB)

            pred_tone_map = cv2.cvtColor(pred_tone_map,
                                         cv2.COLOR_BGR2RGB)

            gt_hdr_name = os.path.join(scene_path,
                                       'gt_hdr.hdr')
            pred_hdr_name = os.path.join(scene_path,
                                       'pred_hdr.hdr')
            gt_tone_map_name = os.path.join(scene_path,
                                        'gt_tone_map.png')           
            tone_map_name = os.path.join(scene_path,
                                        'pred_tone_map.png')

            np.save(os.path.join(scene_path, 'pred_crf'), gen_crf[0,:,:].cpu())
            np.save(os.path.join(scene_path, 'target_crf'), target_crf[0,:,:].cpu())

            cv2.imwrite(pred_hdr_name, denorm_gen_hdr)
            cv2.imwrite(gt_hdr_name, denorm_target_hdr)
            cv2.imwrite(tone_map_name, pred_tone_map)
            cv2.imwrite(gt_tone_map_name, gt_tone_map)

if __name__ == '__main__':
    config = get_config()
    test(config)

