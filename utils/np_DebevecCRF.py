import numpy as np
import torch
from utils.hdr_toolbox import WeightFunction
from utils.hdr_toolbox import LDRStackSubSampling
from utils.hdr_toolbox import FindChromaticyScale
from utils.hdr_toolbox import gsolve

def DebevecCRF(stack,
               stack_exposure,
               nSamples=256,
               sampling_strategy='Grossberg',
               smoothing_term = 20,
               bNormalize =1):

    '''
    This function computes camera response function using Debevec and
    Malik method.

    Input:
        -stack: a stack of LDR images. If the stack is a single or
         double values are assumed to be in [0,1].
        -stack_exposure: an array containg the exposure time of each
         image. Time is expressed in second (s)
        -nSamples: number of samples for computing the CRF
        -sampling_strategy: how to select samples:
          -'Grossberg': picking samples according to Grossberg and
            Nayar algorithm (CDF based)
          -'RandomSpatial': picking random samples in the image
          -'RegularSpatial': picking regular samples in the image
        -smoothing_term: a smoothing term for solving the linear
        -bNormalize: a boolean value for normalizing the inverse CRF
    Output:
        -lin_fun: the inverse CRF
    '''
    # Convert to numpy array, gradient not required
    if not isinstance(stack, np.ndarray):
        stack = np.array(stack.cpu().detach())     
    if not isinstance(stack_exposure, np.ndarray):
        stack_exposure = np.array(stack_exposure.cpu().detach())

    batch_size, num_images, channels, = stack.shape[:3]

    # Log of exposure values
    log_stack_exposure = np.log(stack_exposure)
       
    # Devebec weight function
    W = WeightFunction(np.linspace(0, 1, 256), 'Deb97')
    # Stack sub-sampling
    stack_samples = LDRStackSubSampling(stack, stack_exposure,
                             nSamples, sampling_strategy)
      
    # Recovering camera response function using least method
    g = gsolve(stack_samples, log_stack_exposure,
                  smoothing_term, W)
    lin_fun = np.exp(g)
    # Normalization
    scale = FindChromaticyScale([0.5,0.5,0.5], lin_fun[:,:,127])
    scale = np.expand_dims(scale, 2)
    scale = scale.repeat(nSamples, axis=2)
    scaled_lin_fun = np.zeros((lin_fun.shape))
    for i in range(batch_size):
        scaled_lin_fun[i,:,:] = scale[i,:]*lin_fun[i,:,:]
    
    max_lin_fun = np.amax(lin_fun[:,2,:]) 

    if bNormalize:
        scaled_lin_fun /= max_lin_fun 
    return torch.tensor(scaled_lin_fun).float()
   
