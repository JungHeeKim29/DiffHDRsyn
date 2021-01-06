import numpy as np
import torch
from utils.hdr_toolbox import WeightFunction
from utils.hdr_toolbox import LDRStackSubSampling
from utils.hdr_toolbox import FindChromaticyScale
from utils.hdr_toolbox import MitsunagaNayarCRFClassic

def MitsunagaNayarCRF(stack,
               stack_exposure,
               N = 5,
               nSamples=256,
               sampling_strategy='Grossberg',
               bFull = 1):

    '''
    This function computes camera response function using Debevec and
    Malik method.

    Input:
        -stack: a stack of LDR images. If the stack is a single or
         double values are assumed to be in [0,1].
        -stack_exposure: an array containg the exposure time of each
         image. Time is expressed in second (s)
        -N : order of the polynomial
        -nSamples: number of samples for computing the CRF
        -sampling_strategy: how to select samples:
          -'Grossberg': picking samples according to Grossberg and
            Nayar algorithm (CDF based)
          -'RandomSpatial': picking random samples in the image
          -'RegularSpatial': picking regular samples in the image
    Output:
        -lin_fun: the inverse CRF
        -pp : the coefficients for the polynomial function
    '''
    # Convert to numpy array, gradient not required
    if not isinstance(stack, np.ndarray):
        stack = np.array(stack.cpu().detach())     
    if not isinstance(stack_exposure, np.ndarray):
        stack_exposure = np.array(stack_exposure.cpu().detach())

    batch_size, num_images, channels, = stack.shape[:3]

    # Log of exposure values
    log_stack_exposure = np.log(stack_exposure)
       
    # Stack sub-sampling
    stack_samples = LDRStackSubSampling(stack, stack_exposure,
                                        nSamples, sampling_strategy, 0.01)
      
    stack_samples = stack_samples/255.0

    pp, _ = MitsunagaNayarCRFClassic(stack_samples, 
                                     stack_exposure,N)
    mid_value = 0.5*np.ones((1,channels))
    gray = mid_value
    
    for c in range(channels):
        gray[:,c] = np.polyval(pp[:,c], gray[:,c]) 

    scale = FindChromaticyScale([0.5,0.5,0.5], gray)
    # batch_issue to be solved
    lin_fun = np.zeros((batch_size, channels, nSamples))
    for i in range(batch_size):
        for c in range(channels):
            lin_fun[i,c,:] = scale[i,c] \
                          * np.polyval(pp[:,c], np.linspace(0,1,256))
       
    return torch.tensor(lin_fun).float(), torch.tensor(pp).float()
        
    
