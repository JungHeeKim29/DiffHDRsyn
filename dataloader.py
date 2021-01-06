import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.feature import canny
from PIL import Image
from torch import utils
from torchvision import transforms
from utils.io import *      

class ImageDataset(Dataset):
    def __init__(self, data_dir, img_size=256, 
                 min_ev= int(-3), max_ev=3,
                 fliplr = True, fliptb = True, rot = True, 
                 randcrop = True, transform = True, mode = 'train'):        
        self.scene_num = len(os.listdir(
                          os.path.join(data_dir, mode+'_set')))
        self.img_size = img_size
        self.fliplr = fliplr
        self.fliptb = fliptb
        self.rot = rot
        self.randcrop = randcrop
                 
        self.transform = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize(mean=(0.5, 0.5,0.5), 
                                              std=(0.5,0.5,0.5))])
        self.transform_gray = transforms.Compose(
                              [transforms.ToTensor()])

        self.mode = mode                                             
        self.max_ev = max_ev
        self.min_ev = min_ev       

        # Read images per scene with the type of list 
        # Ex. images[1] : images of scene 1
        
        self.images, self.hdrs = read_stack(data_dir,
                                             min_ev, max_ev, mode)        
        self.evs = torch.tensor([2**i for i in range(min_ev, max_ev+1)])

    def __getitem__(self, idx):
        image_list, hdr_list = self.images[idx], self.hdrs[idx]
        #crf_list = self.crfs[idx]
        
        rand_list = [1]
        if self.rot == True:
            rand_list.append(2)
        if self.fliptb == True:
            rand_list.append(3)
        if self.fliplr == True:
            rand_list.append(4)

        aug_num = np.random.choice(rand_list)

        image_list = [cv2.resize(image.copy(),
                             (self.img_size,self.img_size),
                              interpolation=cv2.INTER_LANCZOS4)\
                       for image in image_list]
        hdr_list = cv2.resize(hdr_list.copy(),
                             (self.img_size, self.img_size),
                             interpolation=cv2.INTER_LANCZOS4)

        if self.randcrop :
            image_list, hdr_list  = self.random_crop(image_list,
                                                     hdr_list,
                                                     self.img_size)

        if aug_num == 1:

            transformed_images = list()
            transformed_hdrs = list()
            transformed_edges = list()

            for i in range(len(image_list)):
                image = image_list[i].copy()
                image = self.cv2_transform(image, self.img_size)
                edge = self.edge_extract(image)
               
                image = self.transform(image)
                transformed_images.append(image)
                edge = self.transform_gray(edge)
                transformed_edges.append(edge)

            hdr = hdr_list.copy()
            hdr = self.hdr_transform(hdr, self.img_size)
            hdr = torch.from_numpy(hdr)
            transformed_hdrs.append(hdr)
         
        if (aug_num==2): 
            rot_angle = np.random.choice([0, 90, 180, 270])

            transformed_images = list()
            transformed_hdrs = list()
            transformed_edges = list()

            for i in range(len(image_list)):
                image = image_list[i].copy()
                image = self.img_rotate(image, rot_angle)
                image = self.cv2_transform(image, self.img_size)

                edge = self.edge_extract(image)

                image = self.transform(image)
                transformed_images.append(image)                

                edge = self.transform_gray(edge)
                transformed_edges.append(edge)

            hdr = hdr_list.copy()
            hdr = self.img_rotate(hdr, rot_angle)
            hdr = self.hdr_transform(hdr, self.img_size)
            hdr = torch.from_numpy(hdr)
            transformed_hdrs.append(hdr)
            
        if aug_num==3 :

            transformed_images = list()
            transformed_hdrs = list()
            transformed_edges = list()

            for i in range(len(image_list)):
                image = image_list[i].copy()
                image = cv2.flip(image, 0)

                image = self.cv2_transform(image, self.img_size)
                edge = self.edge_extract(image)

                image = self.transform(image)
                transformed_images.append(image)    

                edge = self.transform_gray(edge)
                transformed_edges.append(edge)

            hdr = hdr_list.copy()
            hdr = cv2.flip(hdr, 0)
            hdr = self.hdr_transform(hdr, self.img_size)
            hdr = torch.from_numpy(hdr)
            transformed_hdrs.append(hdr)

        if (aug_num==4):

            transformed_images = list()
            transformed_hdrs = list()
            transformed_edges = list()

            for i in range(len(image_list)):
                image = image_list[i].copy()
                image = cv2.flip(image, 1)

                image = self.cv2_transform(image, self.img_size)
                edge = self.edge_extract(image)

                image = self.transform(image)
                transformed_images.append(image)    

                edge = self.transform_gray(edge)
                transformed_edges.append(edge)

            hdr = hdr_list.copy()
            hdr = cv2.flip(hdr, 1)
            hdr = self.hdr_transform(hdr, self.img_size)
            hdr = torch.from_numpy(hdr)
            transformed_hdrs.append(hdr)

        transformed_images = torch.stack(transformed_images, dim=0)     
        transformed_hdrs = torch.stack(transformed_hdrs, dim=0).squeeze(0)
        transformed_edges = torch.stack(transformed_edges, dim=0)

        if np.isnan(np.array(transformed_hdrs).any()):
            print('Nan detected')

        return transformed_images, transformed_hdrs, self.evs, transformed_edges
        
    def __len__(self):
        return self.scene_num

    def cv2_transform(self, image, img_size):
        image = np.array(image,dtype=np.float32)
        image = image/255.0
        return image

    def hdr_transform(self, hdr, hdr_size):
        hdr = np.clip(hdr,1e-6,None)
        hdr = hdr.transpose(2,0,1)
        return hdr

    def img_rotate(self, image, rot_angle): 
        (h, w) = image.shape[:2]
        (cX, cY) = (w/2, h/2)
        M = cv2.getRotationMatrix2D((cX, cY), rot_angle, 1.0)
        rotated = cv2.warpAffine(image, M,(w, h))
        return rotated    

    def edge_extract(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge = canny(gray_img, sigma=2).astype(np.float)
        return np.expand_dims(edge, axis=2)

    def random_crop(self, image_list, hdr_list, crop_size):
         
        image = image_list[0]
        img_height = image.shape[0]
        img_width = image.shape[1]

        r_size = np.random.random()*np.random.randint(1,3)

        while (round(r_size*img_height)<=crop_size) or (
               round(r_size*img_width)<=crop_size) or (
               r_size < 1 and r_size >= 2.05):
               
            r_size= np.random.random()*np.random.randint(1,3)

        cropped_list = list()

        resized_height = round(img_height*r_size)
        resized_width = round(img_width*r_size)

        max_h = resized_height - crop_size
        max_w = resized_width - crop_size

        h = np.random.randint(0, max_h)
        w = np.random.randint(0, max_w)

        for image in image_list: 

            image = cv2.resize(image, (resized_width, resized_height),
                               interpolation=cv2.INTER_CUBIC)

            image = image[h:h+crop_size, w:w+crop_size,:]
            cropped_list.append(image) 

        hdr_list = cv2.resize(hdr_list, (resized_width, resized_height),
                              interpolation=cv2.INTER_CUBIC)

        hdr_list = np.clip(hdr_list, 1e-6, None)
        hdr_list = hdr_list[h:h+crop_size, w:w+crop_size, :]

        return cropped_list, hdr_list
