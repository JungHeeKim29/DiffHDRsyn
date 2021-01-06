import numpy as np
import cv2
import torch
import math
import os
import argparse

from skimage.feature import canny

def read_stack(path, min_ev, max_ev,mode = 'train'):
    # load image            
    data_path = os.path.join(path, mode+'_set')
    target_path = os.path.join(path, mode+'_hdr_set')

    scene_path = sorted([path for path in os.listdir(data_path)
                  if os.path.isdir(os.path.join(data_path, path))])

    scene_target_path = sorted([path for path in os.listdir(target_path)
                         if os.path.isdir(os.path.join(data_path, path))])
    
    total_img_list = list()

    for scene in scene_path :
        img_list =[os.path.join(data_path,
                   scene,
                   scene +'_'+str(i)+'EV_true.jpg.png')
                   for i in range(min_ev, max_ev+1)]

        img_list = [cv2.imread(img_path)
                    for img_path in img_list]
        img_list = [np.array(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
                    for img in img_list]
 
        img_list = np.array(img_list)   #.transpose(0,3,1,2)

        total_img_list.append(img_list)
       
    hdr_list = [os.path.join(target_path, scene_t, 'target_hdr.hdr') 
                for scene_t in scene_target_path]

    hdr_list = [cv2.imread(hdr_path, -1)  
                for hdr_path in hdr_list]
    total_hdr_list = [np.array(
                   cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB))  
                   for hdr in hdr_list]
    
    #crf_list = [os.path.join(target_path, scene_t, 'crf.npy')
    #            for scene_t in scene_target_path]
    #crf_list = [torch.from_numpy(np.load(crf)) for crf in crf_list]
    
    return total_img_list, total_hdr_list #,crf_list

def convert_to_int(array):
    array *= 255

    if type(array).__module__ == 'numpy':
        return array.astype(np.uint8)

    elif type(array).__module__ == 'torch':
        return array.byte()
    else:
        raise NotImplementedError

    
