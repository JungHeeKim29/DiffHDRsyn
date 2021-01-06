import cv2
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.utils import make_grid

def tone_psnr(target, pred, pixel_range = 255):
    mse = np.mean((target-pred)**2)
    return 10*math.log10((pixel_range**2)/mse)

def transform_image(image, img_size) :
    totransform = transforms.Compose([transforms.ToTensor(),
                  transforms.Normalize(mean=(0.5,0.5,0.5), std =(0.5,0.5,0.5))])

    image = cv2.resize(image, (img_size, img_size),
                       interpolation=cv2.INTER_LANCZOS4)
    image = np.array(image, dtype=np.float32)
    image = image/255.0
    image = totransform(image)
    image = image.unsqueeze(0)
    return image

def transform_edge(edge, img_size) :
    totransform = transforms.Compose([transforms.ToTensor(),
                  transforms.Normalize(mean=[0.5], std =[0.5])])

    edge = cv2.resize(edge, (img_size, img_size),
                       interpolation=cv2.INTER_LANCZOS4)
    edge = np.array(edge, dtype=np.float32)
    edge = totransform(edge)
    edge = edge.unsqueeze(0)
    return edge

def compute_psnr(target_img, pred_img, pixel_range=1.0) :
    target_img = ((target_img+1)/2).clamp_(0,1)
    pred_img = ((pred_img+1)/2).clamp_(0,1)
    mse = torch.mean((target_img - pred_img)**2)
    return 10*math.log10((pixel_range**2)/mse)

def transform_hdr(hdr, img_size):
    hdr = cv2.resize(hdr, (img_size, img_size),
                     interpolation = cv2.INTER_LANCZOS4)
    hdr = np.clip(hdr, 0,None)
    hdr = np.array(hdr, dtype=np.float32).transpose(2,0,1)
    hdr = np.expand_dims(hdr, 0)
    return hdr

def inv_transform(image):
    image=image.squeeze(0)
    out = (image+1)/2
    out = out.clamp(0,1)
    return transforms.ToPILImage()(out)

def matplotlib_imshow(img,img2, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = (img+1)/2
    img = img.detach().cpu()
    npimg = img.numpy()
    fig = plt.figure(figsize=(10,10))
    if one_channel :
        plt.imshow(npimg, cmap="Greys")
    else:
        ax1= fig.add_subplot(2, 1,1)
        ax1.imshow(np.transpose(npimg,(1,2,0)))
        ax2 = fig.add_subplot(2,1,2)
        ax2.imshow(img2)
    fig.show()
    plt.pause(0.1)

def tone_map(stack) :
    stack = np.array(stack).squeeze(0).transpose(1,2,0)
    # Tonemap using Reinhard's method to obtain 24-bit color image
    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0)
    ldrReinhard = tonemapReinhard.process(stack.copy())
    return (ldrReinhard*255).clip(0, 255).astype('uint8')

def mu_law(stack, mu=5000.0) :
    numerate = torch.log(1.0+mu*stack)
    denominate = torch.log(torch.tensor(1.0+mu))
    stack = numerate/denominate
    return stack

def extract_hist(x, min_value=0, max_value=256):

    if len(x.shape) > 4:
        x = x.squeeze(0)

    x = (255.0*(x+1.0)/2).clamp(0,255)   

    batch_size, c, h, w = x.shape
    device = x.get_device()

    if x.is_cuda:
        value = torch.arange(min_value, max_value).cuda(device)
        value[0] = 1 
        histogram = torch.zeros((batch_size, c, max_value-min_value)).cuda(device)
    else :

        histogram = torch.zeros((batch_size, c, max_value-min_value))
        value = torch.arange(min_value, max_value)
        value[0] = 1

    value = value.unsqueeze(0).unsqueeze(0)
    value = value.repeat(batch_size, c, 1)

    updates = x.view(batch_size, c, -1)
    indices = updates.int().long()
    histogram = histogram.scatter_add(-1, indices, updates)
    
    return (histogram / value.float())/255

