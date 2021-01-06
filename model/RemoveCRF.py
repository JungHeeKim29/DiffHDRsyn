from .hdr_toolbox import tabledFunction

def RemoveCRF(img, lin_type='gamma',
    lin_fun=2.2):
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
        img_out = tabledFunction(img, lin_fun)
    else:
        img_out = img

    return img_out


