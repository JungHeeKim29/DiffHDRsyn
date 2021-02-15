import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy import io

from utils.np_DebevecCRF import DebevecCRF
from utils.hdr_toolbox import ConstructRad
from utils.np_MitsunagaNayarCRF import MitsunagaNayarCRF

class hdrlayer(nn.Module):
    def __init__(self, 
        merge_type = 'log',
        lin_type = 'LUT',
        method = 'Debevec',
        bMeanWeight = 0,
        weight_type = 'Deb97',
        device = 0,
        num_channels = 3,
        ):

        '''
        Input:
            -merge_type:
              - 'linear': it merges different LDR images in the linear
                 domain.
              - 'log': it merges different LDR images in the natural
                 logarithmic domain.
              - 'w_time_sq': it merges different LDR images in the
                 linear; the weight is scaled by the square of the exposure
                 time.

            -bMeanWeight: if it is set to 1, it will compute a single
            weight for each exposure (not a weight for each color channel)
            for assembling all images.
            Note that this option typicallt improves numerical stability,
            but it can introduce bias in the final colors. This option is
            set to 0 by default.
    '''
        super(hdrlayer, self).__init__()
        self.merge_type = merge_type
        self.bMeanWeight = bMeanWeight
        self.weight_type = weight_type
        self.lin_type = lin_type
        self.device = device
        self.method = method
        # rgb_channel
        # todo: for 1channels
    def forward(self, stack, stack_exposure):
        '''
        Input:
            -stack: an input stack of LDR images. This has to be set if we
            the stack is already in memory and we do not want to load it
            from the disk using the tuple (dir_name, format).
            If the stack is a single or dobule, values are assumed to be in
            the range [0,1].
 
            -stack_exposure: an array containg the exposure time of each
            image. Time is expressed in second (s).

        Output:
           -imgOut: the final HDR image.
           -lin_fun: the camera response function.

        Example:
            This example line shows how to load a stack from disk :
            stack = ReadLDRStack('stack_alignment', 'jpg');      
            stack_exposure = ReadLDRExif('stack_alignment', 'jpg');                 
            BuildHDR(stack, stack_exposure,'tabledDeb97',[],'Deb97');
        '''
        # Correcting the range of image pixel values
        if (stack.view(-1).max() > 255):
           scale = 65535.0 # uint16
        else :
           scale = 1.0
        stack = stack/scale
        batch_size, num_images, channels, height, width = stack.shape

        # Estimate camera response function
        if self.method == 'Debevec':
            lin_fun = DebevecCRF(stack, stack_exposure)
            pp = 0 
        elif self.method == 'Mitsun':
            lin_fun, pp = MitsunagaNayarCRF(stack, stack_exposure, N=5)

        # For numerical stability
        delta_value = 1.0 / 65536.0
        non_zero_offset = 1e-9 
        saturation = 1e-4

        if stack.is_cuda:
            lin_fun = lin_fun.cuda(device= self.device)

        
        #for each LDR image...

        hdr_list = list()

        for i in range(batch_size):        
            stack_b = stack[i,:,:,:,:] # batch stack
            stack_exposure_b = stack_exposure[i, :]
            lin_fun_b = lin_fun[i, :,:] 

            img_out = ConstructRad(stack_b, 
                                   stack_exposure_b, 
                                   lin_fun_b,
                                   pp,
                                   self.weight_type, 
                                   self.merge_type,
                                   self.lin_type,
                                   self.device)                                                        
            hdr_list.append(img_out)
        return torch.stack(hdr_list, dim=0), lin_fun
            
if __name__ == '__main__':
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from scipy import io
    from matplotlib import pyplot as plt
    import os
    def load_image_stack(path):
        scene_path = Path(path)
        stack_list =[os.path.join(path,                   
                   't1' +'_'+str(i)+'EV_true.jpg.png')
                   for i in range(3, -4, -1)]

        print(stack_list)
        img_stack = [np.array(Image.open(path))
            for path in stack_list]
       
        ev = [2**i for i in range(-3,4)]
        return np.array(img_stack).transpose(0, 3, 1, 2), np.array(ev)

    with torch.autograd.set_detect_anomaly(True) :
        path = '/train_set'
        s1, ev1 = load_image_stack(path)
        s1 = (torch.tensor(s1).float() / 255 ).unsqueeze(dim=0).requires_grad_(True)
        ev1 = torch.tensor(ev1).unsqueeze(dim=0)
        test = hdrlayer(method = 'Mitsun')
        result,crf = test(s1, ev1)
        plt.plot(torch.linspace(0,255,256), crf[0,0,:])
        plt.show()

        loss = (result).flatten().abs().mean()
        loss.backward(retain_graph=True)        
        print(s1.grad)
