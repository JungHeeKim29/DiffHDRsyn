import numpy as np
import torch
import torch.nn.functional as F

from scipy import optimize

from torch.autograd import Function
from .sampling import ComputeLDRStackHistogram
from .sampling import GrossbergSampling

def WeightFunction(img, 
    weight_type='Deb97', 
    bounds = [0, 1], bMeanWeight = 0):
    '''
    Input:
        -img: input LDR image in [0,1], 3channel
        -weight_type:
            - 'all': weight is set to 1
            - 'hat': hat function 1-(2x-1)^12
            - 'box': weight is set to 1 in [bounds(1), bounds(2)]
            - 'Deb97': Debevec and Malik 97 weight function
        -bMeanWeight:
        -bounds: range of valid values for Deb97 and box
        -pp:
    Output:
        -weight: the output weight function for a given LDR image
    '''
    if torch.is_tensor(img):
        dtype = 'tensor'
        img = img.cpu().data.numpy()
    elif isinstance(img, np.ndarray):
        dtype = 'numpy'            
    else:
        raise Exception(
              'Data type should be numpy.array or torch.tensor'
)
    if (len(img.shape) > 2) :
       if (img.shape[2] > 1 and bMeanWeight):
           L = torch.mean(img,0).unsqueeze(0)
           img[:,:,:] = L.repeat([3,1,1])
    
    if weight_type == 'all':
        weight = np.ones_like(img)
    elif weight_type == 'box':
        weight = np.ones_like(img)
        index = np.where((img < bound[0])&(img > bound[1]))
        img[index] = 0.0
    elif weight_type == 'hat':
        weight = 1 - (2 * img - 1)**12;
    elif weight_type == 'Deb97':

        Zmin, Zmax = bounds
        tr = (Zmin + Zmax) / 2;
        
        weight = np.zeros_like(img)      
 
        index = np.where(img <= tr)
        weight[index] = img[index] - Zmin

        index = np.where(img > tr)
        weight[index] = Zmax - img[index];
        
        delta = (Zmax-Zmin)/len(weight) 
        
        if delta > 0.0:
            weight = weight / tr;
        weight = np.clip(weight, 0.0, 1.0)

    elif weight_type == 'Robinson':
        shift    = np.exp(-4);
        scaleDiv = (1.0 - shift);
        t = img - 0.5
        weight = (np.exp(-16.0 * (t * t) ) - shift) / scaleDiv;
    else:
        raise NotImplementedError

    if dtype == 'tensor':
       weight = torch.tensor(weight).float()

    return weight

def LDRStackSubSampling(stack, 
    stack_exposure, 
    nSamples=256, 
    sampling_strategy='Grossberg', 
    outliers_percentage = 0):

    '''
    This function subsamples a stack
    Input:
        -stack: a stack of LDR images. If the stack is a single or
                double values are assumed to be in [0,1]
        -nSamples: number of samples for computing the CRF
        -sampling_strategy: how to select samples:
            -'Grossberg'
            -'RandomSpatial'
            -'RegularSpatial'
        -outliers_percentage

    Output:
        -stack_samples: sub-sampled stack
    '''

    #RandomSpatial & RegularSpatial
    #sort_index = np.argsort(stack_exposure)

    # stack sub-sampling
    if sampling_strategy == 'Grossberg':
        stack_hist = ComputeLDRStackHistogram(stack);
        stack_samples = GrossbergSampling(stack_hist, nSamples);

    elif sampling_strategy == 'RandomSpatial':
        raise NotImplementedError
        #stack_samples = RandomSpatialSampling(stack, 
        #     sort_index, nSamples);
        #stack_samples = torch.round(stack_samples * 255);

    elif sampling_strategy == 'RegularSpatial':
        raise NotImplementedError
        #stack_samples = RegularSpatialSampling(stack, 
        #    sort_index, nSamples);
        #stack_samples = torch.round(stack_samples * 255);
    else:
        raise NotImplementedError

    if(outliers_percentage > 0.0):
        t_min = outliers_percentage;
        t_max = 1.0 - t_min;
    
        index = np.where(stack_samples < (t_min * 255.0))
        stack_samples[index] = -1.0
        index = np.where(stack_samples > (t_max * 255.0)) 
        stack_samples[index] = -1.0
    
    return stack_samples

def FindChromaticyScale(M, gray):
    # Color correction
    batch_size, channels  = gray.shape    
    total_scaled_list= list()
    scales = np.zeros((batch_size, 3))
    for i in range(batch_size) :

        l_m = len(M);
        l_I = len(gray);

        def residualFunction(p):
            I_c = gray * p
            I_c_n = I_c / np.linalg.norm(I_c)
            M_n = M / np.linalg.norm(M)
            return ((I_c_n - M_n)**2).sum()
   
        scale = np.ones(l_m)

        min_scale = optimize.fmin(residualFunction, scale, xtol=1e-8,
                  ftol=1e-8, disp=False)
      
        # Shape matching
        scales[i,:] = min_scale
    return scales 

def gsolve(Z_stack, B_stack, l, w) :
    # batch_size, img number, pixel number
    '''
    This function estimates camera response function(CRF)
    from given sample stacks

    Input:
        -Z_stack : a stack of samples from LDR images. 
                  Values are assumed to be in [0,1]
        -B_stack : a stack of exposure value corresponding
                   to the Z_stack
        -l       : smoothing hyper paramter to smoothen 
                   estimated CRF
        -w       : weighting function #Debevec 97

    Output:
        -out     : Camera response function


    '''
    batch_size, num_images, channels,samples = Z_stack.shape
    n = 256
    out = list()
    for Z_3chan, B in zip(Z_stack, B_stack):

        # CRF for each batches
        CRF_b = np.zeros((channels,samples))
        for chan in range(channels):

          # Compute CRF for each channel of Z
          Z = Z_3chan[:,chan,:]

          # Initialize input
          A = np.zeros((Z.shape[-1]*Z.shape[-2] + n + 1, 
                        n + Z.shape[-1]))
          b = np.zeros((A.shape[0], 1))
       
          # Index 

            # Col index for matrix A & b
          index_i = np.arange(n, n + Z.shape[-1]).repeat(Z.shape[-2])
          index_z = Z.transpose(1,0).flatten().astype(np.uint8)
          index_p = np.arange(n-1)

            # Row index for matrix A & b
          index_k1 = np.arange(len(index_z))
          index_k2 = np.arange(len(index_z)+1, len(index_z)+ n)

            # b = w[z]*B[j]    
          B_repeat = np.tile(B, Z.shape[-1])
          
          A[index_k1, index_z] = w[index_z]
          A[index_k1, index_i] = -w[index_z]
          b[index_k1,0] = (w[index_z]* B_repeat)

          # Middle value to be zero(log(1) = 0)
          A[len(index_z), n//2] = 1
 
          # Smoothing term
          A[index_k2, index_p] = l*w[1:]
          A[index_k2, index_p+1] = -2*l*w[1:]
          A[index_k2, index_p+2] = l*w[1:]

          # Solve least square prob
          x = np.linalg.lstsq(A, b, rcond = None)[0]
          #x =  np.dot(np.linalg.pinv(A),b)

          CRF_b[chan, :] = x[:n].squeeze()

        out.append(CRF_b)

    return np.stack(out, axis=0) # batch_size, channels, samples

# --------------------Debevec numpy tool------------------------- 

def MitsunagaNayarCRFClassic(stack_samples, stack_exposure, N):
    
    threshold = 1e-4
    batch = stack_samples.shape[0]
    chan = stack_samples.shape[2]
    Q = stack_exposure.shape[1]
    maxiterations = -1
    Mmax = 1.0
    def MN_d(c,q,n):
        q_p = q + 1
        M_q = stack_samples[:,q,c,:]
        M_q_p = stack_samples[:, q_p,c,:]

        M_q_t = M_q[(M_q>0) &(M_q_p>0)]
        M_q_p_t = M_q_p[(M_q>0) & (M_q_p>0)]
        d = np.power(M_q_t,n) -R[:,q]*np.power(M_q_p_t,n)

        return d.ravel()
    pp = np.zeros((N+1, chan))
    pp_prev = np.zeros((N+1, chan))

    err = 0

    R0 = np.zeros((batch, Q-1))

    for q in range(Q-1) :
        R0[:,q] = stack_exposure[:,q]/stack_exposure[:, q+1]

    x = [i/255.0 for i in range(255)]

    for c in range(chan):
        R = R0
        bLoop =1
        iter_n = 0
        while(bLoop):
            A = np.zeros((N,N))
            b = np.zeros((N,1))
            for i in range(N):
                for j in range(N):
                    for q in range(Q-1):
                        delta = MN_d(c,q,j)-MN_d(c,q,N)
                        A[i,j] = A[i,j] + \
                               sum(np.multiply(MN_d(c,q,i), delta))
             
                for q in range(Q-1):
                    b[i] = b[i] - \
                    sum(np.multiply(Mmax*MN_d(c,q,i), MN_d(c,q,N)))

            coeff = np.linalg.lstsq(A, b, rcond = None)[0]
            coeff_n = Mmax - sum(coeff)

            pp[:, c] = np.flip(np.append(coeff,coeff_n))

            f_1 = np.polyval(pp[:,c], x)
            f_2 = np.polyval(pp_prev[:,c], x)
            bLoop = max(np.abs(f_1-f_2) > threshold)
            bLoop = bLoop and (iter_n < maxiterations)
            if (bLoop):
                pp_prev = pp
                for q in range(Q-1):
                    s1 = stack_samples[:,q,c,:]
                    s2 = stack_samples[:,q+1,c,:]
                    indx = (s1>0)&(s2>0)
                    e1 = np.polyval(pp[:,c], s1[indx])
                    e2 = np.polyval(pp[:,c], s2[indx])
                    R[:,q] = sum(e1)/sum(e2)

                iter += 1
        for q in range(Q-1):
            s1 = stack_samples[:, q, c]
            s2 = stack_samples[:, q+1, c]             
            indx = (s1>0) & (s2>0)
            e1 = np.polyval(pp[:,c], s1[indx])
            e2 = np.polyval(pp[:,c], s2[indx])
            err += sum((e1-R[:,q]*e2)**2)

    return pp, err

# ------------------------MitsunagaNayarCRF ------------------------

def tabledFunction(img, table, device):

    '''
    Returns remapped values regarding the table function
 
    Input:
        -img: an LDR image or stack with values in [0,2^nBit - 1]
        -table: three functions for remapping image pixels values
    Output:
        -img: an LDR image with remapped values
    '''

    plf = Piecewise_Linear.apply
    img_out = torch.zeros_like(img)
    non_zero_offset = 1e-7*torch.ones_like(table)
    table = torch.where(table>0, table, non_zero_offset) 
    if len(img.shape) > 3 :
        num_images, channels = img.shape[:2]
        for i in range(num_images):
            for j in range(channels): 
                img_out[i,j,:,:] = plf(img[i,j,:,:], table[j,:], device)
       
    else :
        channels = img.shape[0]
        for j in range(channels):
            img_out[j,:,:] = plf(img[j,:,:], table[j,:], device)
        
    return img_out

def polyFunction(img, table, pp, device):
    poly = Polynomial.apply
    img_out = torch.zeros_like(img)
    non_zero_offset = 1e-7*torch.ones_like(img)
    if len(img.shape) > 3 :
        num_images, channels = img.shape[:2]
        for i in range(num_images):
            for j in range(channels):
                img_out[i,j,:,:] = poly(img[i,j,:,:],table[j,:], pp[:,j], device)

    else :
        channels = img.shape[0]
        for j in range(channels):
            img_out[j,:,:] = poly(img[j,:,:], table[j,:],pp[:,j], device)

    return torch.where(img_out>0, img_out, non_zero_offset)

def RemoveCRF(img, lin_type, lin_fun,
    pp, device = 1):

    '''
    This function builds an HDR image from a stack of LDR images.
    Input:
        -img: an image with values in [0,1]
        -lin_type: the linearization function:
           - 'linear': images are already linear
           - 'gamma2.2': gamma function 2.2 is used for linearization
           - 'sRGB': images are encoded using sRGB
           - 'LUT': the lineraziation function is a look-up
                    table defined stored as an array in the
                    lin_fun
        -lin_fun: it is the camera response function of the camera that
           took the pictures in the stack. If it is empty, [], and
           type is 'LUT' it will be estimated using Debevec and Malik's
           method.
    Output:
        -imgOut: a linearized (CRF is removed) image.
    '''

    # linearization of the image
    if lin_type == 'gamma':
        img_out = img ** lin_fun
    elif lin_type == 'sRGB':
        raise NotImplementedError
        #imgOut = ConvertRGBtosRGB(img, 1);
    elif lin_type == 'LUT':
        img_out = tabledFunction(img, lin_fun, device)
    elif lin_type == 'poly':
        img_out = polyFunction(img, lin_fun, pp, device) 

    return img_out

class Piecewise_Linear(Function):
    @staticmethod
    def forward(ctx, x, w, device):
        x = torch.round(x*255)
        x = x.type(torch.uint8)
        index_x = x.flatten().long()
        slope = [index_x[0]] + [i-j for i, j in zip(w[1:], w)]
        if x.is_cuda:
            slope = torch.tensor(slope).float().cuda(device=device)
            piecewise = w.float().cuda(device=device)
        else:
            slope = torch.tensor(slope).float()
            piecewise = w.float()

        slope_x = slope[index_x.long()].reshape(x.shape)

        result = piecewise[x.long()]
        ctx.save_for_backward(slope_x)


        return result

    @staticmethod
    def backward(ctx, grad_output):
       
        result, = ctx.saved_tensors
        return grad_output * result, None, None

class Polynomial(Function):
    @staticmethod
    def forward(ctx, x, w, pp, device):
        x = torch.round(x*255)
        x = x.type(torch.uint8)
        index_x = x.flatten().long()
        der_poly = torch.tensor([pp[i]*i for i in range(pp.shape[0])]) 
        slope = torch.tensor(np.polyval(der_poly, torch.linspace(0,1,256)))
        if x.is_cuda:
            slope = slope.float().cuda(device=device)
            poly = w.float().cuda(device=device)
        else:
            slope = slope.float()
            poly = w.float()
        
        slope_x = slope[index_x.long()].reshape(x.shape)

        result = poly[x.long()]
        ctx.save_for_backward(slope_x)
        return result

    @staticmethod
    def backward(ctx, grad_output):

        result, = ctx.saved_tensors
        return grad_output * result, None, None, None  
    
def ConstructRad(stack,
                 exposure_stack,
                 lin_fun,
                 pp,
                 weight_type='Deb97',
                 merge_type='log',
                 lin_type = 'LUT',
                 device = 0):

    num_images, channels, height, width = stack.shape
    # Initialization
    if stack.is_cuda==True:
        mask_zeros = torch.zeros(channels, height, width).cuda(device=device)
        mask_ones  = torch.ones(channels, height, width).cuda(device=device)
    else:
        mask_zeros = torch.zeros(channels, height, width)
        mask_ones  = torch.ones(channels, height, width)

    med = len(exposure_stack)//2
    img_med = torch.clamp(stack[med,:,:,:], 0.0, 1.0)

    # For saturated pixels
    sat = torch.argmin(exposure_stack)

    img_sat = torch.clamp(stack[sat, :,:,:], 0.0, 1.0)
    img_sat = RemoveCRF(img_sat, lin_type, lin_fun,pp,device)
    img_sat = img_sat/exposure_stack[sat]

    # For noisy pixels
    noisy = torch.argmax(exposure_stack)

    img_noisy = torch.clamp(stack[noisy, :, :, :], 0.0, 1.0)
    img_noisy = RemoveCRF(img_noisy, lin_type, lin_fun,pp,device)
    img_noisy = img_noisy/exposure_stack[noisy]

    # Offsets for numerical stability(Avoid zero division)
    delta_value = 1.0/65536.0
    non_zero_offset = 1e-9
    saturation =  1e-4
   
    # Normalizing stack
    clamped_stack = torch.clamp(stack, 0.0, 1.0)
    if stack.is_cuda :
        weight = WeightFunction(clamped_stack, weight_type).cuda(device=device)
        rm_crf_stack = RemoveCRF(clamped_stack, lin_type, lin_fun,pp,device)
    else: 
        weight = WeightFunction(clamped_stack, weight_type)
        rm_crf_stack = RemoveCRF(clamped_stack, lin_type, lin_fun, pp)

    # Merging stack images
    if merge_type =='linear' :
        raise NotImplementedError
        #imgOut = imgOut + (weight * tmpStack) / dt_j
        #totWeight = totWeight + weight;

    elif merge_type == 'log':
        img_chan = list()
        for n in range(num_images):
            img_temp = weight[n,:,:,:]*(torch.log(rm_crf_stack[n,:,:,:])-torch.log(exposure_stack[n]))         
            img_chan.append(img_temp)
        img_out = torch.stack(img_chan, dim=0)
        img_out = img_out.sum(dim=0)
        totWeight = weight.sum(dim=0)

    elif merge_type == 'w_time_sq':
        raise NotImplementedError
        # Todo
        #imgOut = imgOut + (weight * tmpStack) * dt_j
        #totWeight = totWeight + weight * dt_j * dt_j

    # Output images
    totWeight +=non_zero_offset
    img_out = img_out/(totWeight)
    img_out = torch.exp(img_out)

    # Masking zero division outliers       
    mask_sat = torch.where((totWeight<=saturation)&(img_med > 0.5),
                         mask_ones, mask_zeros)
    mask_sat = mask_sat.sum(0).repeat(3,1,1)

    mask_noisy = torch.where((totWeight<=saturation)&(img_med < 0.5),
                         mask_ones, mask_zeros)
    mask_noisy = mask_noisy.sum(0).repeat(3,1,1)

    img_out = torch.where(mask_sat>0, img_sat, img_out)
    img_out = torch.where(mask_noisy>0, img_noisy, img_out)

    return img_out    
  
plf = Piecewise_Linear.apply
poly = Polynomial.apply

if __name__ == '__main__':
    
    w = WeightFunction(np.linspace(0,1,256), 'Deb97')
    w = WeightFunction(np.linspace(0,1,256), 'all')
    w = WeightFunction(np.linspace(0,1,256), 'box')
    w = WeightFunction(np.linspace(0,1,256), 'hat')
    w = WeightFunction(np.linspace(0,1,256), 'Robinson')

