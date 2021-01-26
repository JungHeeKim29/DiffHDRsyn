import numpy as np

def ComputeLDRStackHistogram(stack):
    '''
    Input: stack
    Output:
        -stackOut: a stack of LDR image histograms
    '''
    batch_size, num_images, channels, height, width = stack.shape
    stackOut = np.zeros((batch_size, num_images, channels, 256))  

    for i in range(batch_size):
        for j in range(num_images):
            for k in range(channels):
                tmp = stack[i, j, k, :, :]
                tmp = np.clip(np.round(tmp * 255), 
                    0.0, 255.0)
                stackOut[i, j, k, :] = np.histogram(tmp, bins=256,
                    range=(0.0,255.0))[0]
              
    return stackOut

def GrossbergSampling(stack, nsamples = 256):
    '''
    Input:
        -stack: a stack of LDR histograms; 
        -nSamples: the number of samples for sampling the stack

    Output:
        -stackOut: a stack of LDR samples for Debevec and Malik method
    '''
    batch_size, stackSize, channels, _ = stack.shape
   
    for i in range(batch_size):
        for j in range(stackSize):
            for k in range(channels):
                h_cdf = np.cumsum(stack[i,j,k,:], axis=0)
                stack[i,j,k, :] = h_cdf / h_cdf.max()
    u = np.linspace(0.0, 1.0, nsamples)
    stackOut = np.zeros((batch_size, stackSize, channels, nsamples))

    for l in range(len(u)):
        for i in range(batch_size):
            for j in range(stackSize):
                for k in range(channels): 
                    #for l in range(len(u)):
                    val = np.argmin(np.abs(stack[i,j,k,:] - u[l]))
                    stackOut[i,j,k,l] = val
    return stackOut
