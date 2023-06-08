import numpy as np
from typing import List
import matplotlib.pyplot as plt 
import torch
from torch import nn
from torch.nn import Conv2d
from fase.core.seal_ckks import SEALContext
from fase.core.commonAlgo import CommonAlgorithms
from fase.seal import Ciphertext

def left_rotate(arr: np.ndarray, r: int) -> np.ndarray:
    return np.roll(arr, -r)

def repeat_filter(kernel: np.ndarray, n_repeat: int) -> np.ndarray:
    return np.broadcast_to(kernel,(n_repeat,)+kernel.shape)
    
def annot_heatmap(ax: plt.Axes, img: np.ndarray) -> None:
    ax.imshow(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            text = ax.text(j, i, img[i, j],
                           ha="center", va="center", color="w")

######################
def rotate_input(img: np.ndarray, kernel_size:tuple):
    if img.ndim == 2:
        nx, ny = img.shape
        imgarr = img.ravel()
    elif img.ndim == 3:
        nx, ny, nc = img.shape
        imgarr = img.ravel()
    f_w, f_h = kernel_size
    
    rotated = []
    for i in range(f_w):
        for j in range(f_h):
            rotated.append(left_rotate(imgarr, (i -1)*nx + (j-1)))
    return rotated


def get_out_size(img_size: tuple, 
                 kernel_size: tuple, 
                 stride: int =1, 
                 padding: str ='same'):
    """calculate convolution output image size 
    
    Image is assumed to be 2D for now (n_batch, nx, ny, n_channel = img.shape, later)
    """    
    nx_i, ny_i = img_size
    f_w, f_h = kernel_size
    s_w = s_h = stride
    
    if padding == "valid":
        # Use only valid pixels without padding
        nx_o = (nx_i - f_w + 1)/s_w
        ny_o = (ny_i - f_h + 1)/s_h
    elif padding == "same":
        # pad so that the output shape doesn't change if not for stride
        nx_o = nx_i/s_w
        ny_o = ny_i/s_h

    nx_o = int(np.ceil(nx_o))
    ny_o = int(np.ceil(ny_o))
    
    return nx_o, ny_o


#########################################################
###################### custom conv#######################
#########################################################


def rotate_a_channel(img: np.ndarray, f_h: int, f_w: int):
    """Make sure order of nh,nw & f_h,f_w match
    """
    nh, nw = img.shape
    imgarr = img.ravel()

    rotated = []
    for i in range(f_h):
        for j in range(f_w):
            rotated.append(left_rotate(imgarr, (i -1)*nh + (j-1)))
    
    return rotated

def rotate_input_channels(img_tensor_a_batch:np.ndarray,
                          f_h: int, f_w: int):
    rotated_channels = []
    for imgarr in img_tensor_a_batch:
        rotated_channels.append(rotate_a_channel(imgarr, f_h, f_w))
        
    return rotated_channels

def rotate_a_batch(img_batch:np.ndarray, kernel_shape:tuple):
    """should I handle a batch of images at a time?
    """
    #img_np = img_tensor_a_batch.numpy()
    nb, nc, nh, nw = img_batch.shape
    c_out, c_in, f_h, f_w = kernel_shape
    assert nc == c_in
    
    rotated_batch=[]
    for this_img in img_batch:
        rotated_batch.append(rotate_input_channels(this_img, f_h, f_w))
        
    return rotated_batch


def my_conv2D(img_tensor:np.ndarray, kernel:np.ndarray):
    """
    Note
    ----
    converting a list to a tensor is 'extremely' slow. 
    So I convert the list to a numpy array first.
    """
    c_out, c_in, f_h, f_w = kernel.shape
    n_batch, n_channel, nh, nw = img_tensor.shape
    
    rotated = rotate_a_batch(img_tensor, kernel.shape)
    out_nh, out_nw = get_out_size((nh, nw), (f_h, f_w), stride=1, padding='same')

    conv_out=[]
    for this_example in rotated:
        result_each_img =[]
        for this_kernel_in_channel in kernel:
            # from 0 - 7
            # multi-channel conv sum

            result_each_out_channel = np.zeros(out_nh * out_nw, dtype=np.float32)
            for this_channel, this_kernel in zip(this_example, this_kernel_in_channel):
                # from 0 - 2

                ####### single 2D img convolution
                rkernel = this_kernel.ravel() # (f_h * f_w)
                for rr, rk in zip(this_channel, rkernel):
                    # it can simply be done a as vectorized multiplication.
                    # But keeping this form to better relate it with FHE version.
                    result_each_out_channel += rr * rk

            result_each_img.append(result_each_out_channel.reshape(out_nh, out_nw))
        conv_out.append(np.stack(result_each_img))
    return np.stack(conv_out)


#########################################################
######################### FHE ###########################
#########################################################


def aug_ctxt(ctxt: Ciphertext, nh: int, nw:int, sec:SEALContext) -> Ciphertext:
    """(Obsolete) repeat ctxt before and after to mimic the circulation of an array rotation
    """
    ctxt_aug = sec.add(ctxt, sec._evaluator.rotate_vector(ctxt, -nh*nw, sec.galois_keys))
    sec.add(ctxt_aug, sec._evaluator.rotate_vector(ctxt, nh*nw, sec.galois_keys), inplace=True)
    return ctxt_aug

def rotate_for_conv(ctxt: Ciphertext, 
                    nw:int, 
                    f_h:int, 
                    f_w:int, 
                    sec:SEALContext, 
                    stride:int =1)->List:
    rotated_ctxt = []
    hhw = int((f_w -1)/2)
    hhh = int((f_h -1)/2)
    for i in range(f_w)[::stride]:
        for j in range(f_h)[::stride]:
            #print("Rotating...", f_w, f_h,  i, j, (i-hhh)*nw + (j-hhw))
            rotated_ctxt.append(sec._evaluator.rotate_vector(ctxt, (i-hhh)*nw + (j-hhw), sec.galois_keys))
    return rotated_ctxt

def gen_pad_mask(kernel_size, img_size, pad=1):
    kh, kw = kernel_size
    nx, ny = img_size
    img_len = nx*ny

    masks = []
    for ki in range(kh):
        for kj in range(kw):
            mask = np.ones(img_len).reshape(nx,ny)
            if kj == 0:
                mask[:,0] = 0
            if kj == kw-1:
                mask[:,-1] = 0
            if ki == 0:
                mask[0,:] = 0
            if ki == kh-1:
                mask[-1,:] = 0
            
            masks.append(mask.ravel())
    return masks

def convolve_fhe(rotated_ctxt:List, 
                kernel:np.ndarray, 
                sec:SEALContext, 
                pad_masks:List=None,
                eps:float=1e-6,
                ) -> List:
    """
    per-channel convolution.

    params
    ------
    rotated_ctxt: 
        list of rotated ctxts of a channel

    ignores multiplciation with vales less than eps. (assumed to be 0)
    
    NOTE
    ----
    When the input data is normalized [0,1], non-negligible number of weights 
    are smaller than 1e-6.
    """
    if pad_masks is None: pad_masks = [False] * len(rotated_ctxt)

    rkernel = kernel.ravel()
    out_fhe = None
    for rk, rr, mask in zip(rkernel, rotated_ctxt, pad_masks):
        #print("mask", mask)
        if abs(rk) > eps:
            if out_fhe is None:
                out_fhe = sec.multByConst(rr, rk, broadcast=True, rescale=True)
                if mask is not False:
                    mm = np.zeros(int(out_fhe.poly_modulus_degree()/2))
                    mm[:len(mask)] = mask
                    sec.multByConst(out_fhe, mm, inplace=True, rescale=True)
            else:
                tmp = sec.multByConst(rr, rk, broadcast=True, rescale=True)
                if mask is not False:
                    mm = np.zeros(int(tmp.poly_modulus_degree()/2))
                    mm[:len(mask)] = mask
                    sec.multByConst(tmp, mm, inplace=True, rescale=True)
                sec.add(out_fhe, tmp, inplace=True)
                print(sec.decrypt(out_fhe)[:1024])

    #print(sec.decrypt(out_fhe)[:1024])
    return out_fhe

def my_conv2D_FHE(sec:SEALContext, 
                img_enc:List, 
                nh:int, 
                nw:int, 
                convlayer:Conv2d,
                stride_in:int =1, 
                stride_out:int =1, 
                padding:str ="same"):
    """
    
    Convolution between encrypted image and plain kernel

    parameters
    ----------
    sec : Ciphertext
    stride_in: strides that have applied to img_enc so far
    stride_out: stride additionaly applied to img_enc this time
    kernel: conv layer weights from a Pytorch model
    nh: height of the **original** image
    nw: width of the **original** image
    
    Note
    ----
    1. converting a list (of list) to a tensor is 'extremely' slow. 
       So, convert the list to a numpy array first.
    2. strided convolution results in sparse ciphertext. 
       Strides must be tracked to determine value slots of a ctxt.
       It's important to distinguish *strides so far* and the *new stride*.
    
    """
    kernel = convlayer.weight.detach().numpy()
    bias = convlayer.bias.detach().numpy()

    c_out, c_in, f_h, f_w = kernel.shape
    # e.g., new stride_2 to strided_2 = stride_4
    stride_out *= stride_in
    out_nh, out_nw = get_out_size((nh, nw), (f_h, f_w), 
                                  stride=stride_out, 
                                  padding=padding)
    
    print("Output image size", out_nh, out_nw)
    
    # 3 - > 5, 5 - > 9 , ... 
    dilated_shape = ((f_h-1)*stride_in+1,
                     (f_w-1)*stride_in+1,)
    #dilated_kernel = np.zeros(dilated_shape)

    # Striding by using mask
    mask = np.zeros(int(sec.parms.poly_modulus_degree()/2))
    if stride_out == 1:
        mask[:nw*nh] = np.ones(nw*nh)
    else :
        _mask = np.zeros((nw,nh))
        _mask[::stride_out,::stride_out] = 1.
        mask[:nw*nh] = _mask.ravel()

    #print(mask[:1024].reshape(32,32))
    # padding mask
    pad_masks = gen_pad_mask((f_h, f_w), (nh, nw))

    # rotate each channel
    rotated =[]
    for channel_enc in img_enc:
        #img_aug = aug_ctxt(channel_enc, nh, nw, sec)
        rotated.append(rotate_for_conv(channel_enc, nw, 
                                        dilated_shape[0],
                                        dilated_shape[1], 
                                        sec, 
                                        stride=stride_in))

    ### Main loop ###
    conv_out=[]
    for this_kernel_in_channel, this_bias in zip(kernel, bias):

        # convolve each channel of image and kernel
        result_each_out_channel = None
        for this_channel, this_kernel in zip(rotated, this_kernel_in_channel):
            # 0 - c_in
            if result_each_out_channel == None:
                result_each_out_channel = convolve_fhe(this_channel, this_kernel, sec, pad_masks)
            else:
                sec.add(result_each_out_channel,
                        convolve_fhe(this_channel, this_kernel, sec, pad_masks), inplace=True)
        sec.addConst(result_each_out_channel, this_bias*mask, inplace=True)


        conv_out.append(result_each_out_channel)

    return conv_out, out_nh, out_nw


def __incomplete__my_conv2D_FHE1x1(sec, ctxts_in, nh, nw, kernel, stride=1, padding="same"):
    """Simpler convolution when kernel evaluations don't overlap with each other

    todo
    ----
    support for strided ctxt

    """
    if torch.is_tensor(kernel):
        kernel = kernel.detach().numpy()

    c_out, c_in, f_h, f_w = kernel.shape
    
    out_nh, out_nw = get_out_size((nh, nw), (f_h, f_w), 
                                  stride=stride, 
                                  padding=padding)
    
    print("Output image size", out_nh, out_nw)

    mask = np.zeros(int(sec.parms.poly_modulus_degree()/2)) # No int!
    _mask = np.zeros((nw,nh))

    conv_out=[]
    for ii, this_kernel_in_channel in enumerate(kernel):
        # 0 - c_out
        # multi-channel conv sum

        # convolve each channel of image and kernel
        result_each_out_channel = None
        for jj, (this_channel, this_kernel) in enumerate(zip(ctxts_in, this_kernel_in_channel)):
            _mask[::stride,::stride] = this_kernel.squeeze()
            mask[:nw*nh] = _mask.ravel()
            # 0 - c_in
            if result_each_out_channel == None:
                result_each_out_channel = sec.multByConst(this_channel, mask, broadcast=False, inplace=False, rescale=True)
            else:
                sec.add(result_each_out_channel,
                        sec.multByConst(this_channel, mask, broadcast=False, inplace=False, rescale=True),
                        inplace=True)
        conv_out.append(result_each_out_channel)
    return conv_out, out_nh, out_nw


def fhe_bn(sec, ctxts, torch_bn, eps = 1e-5):
    """batch normalization of homomorphic CNN
    
    """
    gamma = torch_bn.weight.detach().cpu().numpy()
    beta = torch_bn.bias.detach().cpu().numpy()
    running_mean = torch_bn.running_mean.detach().cpu().numpy()
    running_var = torch_bn.running_var.detach().cpu().numpy()
    
    # Per channel
    result = []
    slot_count = int(ctxts[0].poly_modulus_degree()/2)
    for i, this_channel in enumerate(ctxts):
        denom = np.sqrt(running_var[i] + eps)
        factor = gamma[i]/denom
        const  = -running_mean[i]/denom*gamma[i] + beta[i]

        sec.multByConst(this_channel, np.repeat(factor, slot_count), inplace=True, rescale=True)
        sec.addConst(this_channel, np.repeat(const, slot_count), inplace=True)

        result.append(this_channel)
        
    return result


def rotate_for_pool(ctxt: Ciphertext, 
                    nw:int, 
                    f_h:int, 
                    f_w:int, 
                    sec:SEALContext, 
                    stride:int =1)->List:
    """
    Tested only for 2x2, stride=2 pooling
    """

    rotated_ctxt = []
    for i in range(f_w)[::stride]:
        for j in range(f_h)[::stride]:
            rotated_ctxt.append(sec._evaluator.rotate_vector(ctxt, i*nw + j, sec.galois_keys))
    return rotated_ctxt


def fhe_avg_pool(sec, ctxts, nh, nw, stride_in, kernel_size=2, padding = "same"):
    """
    Average pooling in exact striding.

    Note
    ----
    Different rotation pattern than padding="same" convolution.


    """
    stride_out = f_h = f_w = kernel_size
    # e.g., new stride_2 to strided_2 = stride_4
    stride_out *= stride_in
    out_nh, out_nw = get_out_size((nh, nw), (f_h, f_w), 
                                  stride=stride_out, 
                                  padding=padding)

    print("Output image size", out_nh, out_nw)

    dilated_shape = ((f_h-1)*stride_in+1,
                     (f_w-1)*stride_in+1,)

    kernel = np.zeros((kernel_size, kernel_size))
    kernel[:,:] = 1./(kernel_size**2)
    
    #dilated_kernel = np.zeros(dilated_shape)
    #dilated_kernel[::stride_in,::stride_in] = 1./(kernel_size**2)

    # rotate each channel
    rotated =[]
    for channel_enc in ctxts:
        #img_aug = aug_ctxt(channel_enc, nh, nw, sec)
        rotated.append(rotate_for_pool(channel_enc, nw, 
                                       dilated_shape[0],
                                       dilated_shape[1], 
                                       sec, 
                                       stride=stride_in))


    # Striding uses mask
    mask = np.zeros(int(sec.parms.poly_modulus_degree()/2))

    _mask = np.zeros((nw,nh))
    _mask[::stride_out,::stride_out] = 1. # to match torch.Avgpool2D behavior
    mask[:nw*nh] = _mask.ravel()

    # padding mask
    #pad_masks = gen_pad_mask((f_h, f_w), (nh, nw))
    output = []
    for this_channel in rotated:
        tmp = convolve_fhe(this_channel, kernel, sec, None)
        output.append(sec.multByConst(tmp, mask, broadcast=False, inplace=False, rescale=True))
    return output, out_nh, out_nw

##############################################
###  Temporary
##############################################
def do_bootstrap(sec, ctxt):
    ctxt = sec.decrypt(ctxt)
    return sec.encrypt(ctxt)

poly_mult_depth = 5

def approx_relu_fhe(sec, calgo, ctxts, ff, xfactor = 20, repeat = 3):
    output = []
    for ictx, ctx in enumerate(ctxts):
        #tmp1 = sec.decrypt(ctx)
        #print("Org range",tmp1.min(), tmp1.max())
        #print("factor", xfactor)

        scaled = sec.multByConst(ctx, 1/xfactor, rescale=True, broadcast=True, inplace=False)

        tmp = calgo.function_poly(ff.coef, scaled)

        for _ in range(1,repeat):
            if sec.context.get_context_data(tmp.parms_id()).chain_index() < poly_mult_depth:
                tmp = do_bootstrap(sec, tmp)
                print("Bootstrapped", ictx, _)
            tmp = calgo.function_poly(ff.coef, tmp)
            #tmp = calgo.function_poly(ff.coef, tmp)

        # (out + 1)
        sec.addConst(tmp, np.ones(1024), inplace=True)
        # (out + 1) / 2
        sec.multByConst(tmp, 0.5, inplace=True, rescale=True, broadcast=True)

        #Mod mismatch between ctx and tmp
        l_ctx = sec.context.get_context_data(ctx.parms_id()).chain_index()
        l_tmp = sec.context.get_context_data(tmp.parms_id()).chain_index()
        if l_tmp > l_ctx:
            sec.match_mod(tmp, ctx)
        elif l_ctx > l_tmp:
            sec.match_mod(ctx, tmp)

        # x * (out + 1) /2
        sec.mult(tmp, ctx, inplace=True)
        sec.rescale(tmp)

        output.append(tmp)
    return output

 ###################################   




def strided_indices(nx, ny, strx, stry):
    """return indices of valid pixels in a strided ctxt img.
    
    
    example
    -------
    if nx = 8, ny = 8, and strride = 2,
    valid pixels are:
    
    1 0 1 0 1 0 1 0 
    0 0 0 0 0 0 0 0
    1 0 1 0 1 0 1 0 
    0 0 0 0 0 0 0 0
    1 0 1 0 1 0 1 0 
    0 0 0 0 0 0 0 0
    1 0 1 0 1 0 1 0 
    0 0 0 0 0 0 0 0
    
    or in a ctxt, 
    
    1 0 1 0 1 0 1 0, 0 0 0 0 0 0 0 0, 1 0 1 0 1 0 1 0, ...
    
    then the function returns 
    
    [0,2,4,6,16,18,20, ...]
    
    """
    # valid image
    nx_v = int(nx/strx)
    ny_v = int(ny/stry)

    valid_inds = []
    for i in range(nx_v):
        for j in range(ny_v):
            valid_inds.append(nx*i*strx + stry*j)
    return valid_inds

def reshape(sec, ctxts):
    # tmp10
    nx = ny = 32

    ind_put = 0
    ind_source = strided_indices(nx, ny, 8, 8)
    for ich, tt in enumerate(ctxts):
        for ind_ch in ind_source:
            if ind_put == 0:
                mask = np.zeros(sec.nslots)
                mask[ind_ch] = 1.
                output = sec.multByConst(tt, mask, rescale=True)

            mask = np.zeros(sec.nslots)
            mask[ind_ch] = 1.
            masked = sec.multByConst(tt, mask, rescale=True)

            sec.lrot(masked, ind_ch - ind_put, inplace=True)
            #util.check_ctxt(masked)
            #util.check_ctxt(output)
            sec.add(output, masked, inplace=True)
            ind_put += 1
        #print(ind_put)

    return output
    # 1 x 1024 

def fullyConnected1(sec, ctxt: Ciphertext, fc1: nn.modules.Linear):
    """Compute a FullyConnected layer
    
    parameters
    ----------
    ctxt: single Ciphertext carrying a n-dim feature in compact packing
    
    
    """
    calgo = CommonAlgorithms(sec, "SEAL")
    tmp_fc1 = sec.encrypt(np.zeros(sec.nslots)) 


    ind_put = 0
    for ww in fc1.weight.detach().numpy():
        feature_weight = sec.multByConst(ctxt, ww, rescale=True)

        # Sum
        reduced = calgo.reduce(feature_weight, 11)

        mask = np.zeros(sec.nslots)
        mask[ind_put] = 1.
        masked = sec.multByConst(reduced, mask, rescale=True)

        if ind_put == 0: sec.match_mod(tmp_fc1, masked)
        sec.add(tmp_fc1, masked, inplace=True)        
        ind_put += 1


    bias = np.zeros(sec.nslots)
    bias[:128] = fc1.bias.detach().numpy()

    sec.addConst(tmp_fc1, bias, inplace=True)
    
    return tmp_fc1


def fullyConnected2(sec, ctxt: Ciphertext, fc2: nn.modules.Linear):
    """Compute a FullyConnected layer
    
    parameters
    ----------
    ctxt: single Ciphertext carrying a n-dim feature in compact packing
    
    
    """
    calgo = CommonAlgorithms(sec, "SEAL")
    # But only 10 slots will be used
    tmp_fc2 = sec.encrypt(np.zeros(sec.nslots)) 

    ind_put = 0
    for ww in fc2.weight.detach().numpy():
        feature_weight = sec.multByConst(ctxt, ww, rescale=True)

        # Sum
        reduced = calgo.reduce(feature_weight, 7) # 7 or 8?

        mask = np.zeros(sec.nslots)
        mask[ind_put] = 1.
        masked = sec.multByConst(reduced, mask, rescale=True)

        if ind_put == 0: sec.match_mod(tmp_fc2, masked)
        sec.add(tmp_fc2, masked, inplace=True)
        ind_put += 1


    bias = np.zeros(sec.nslots)
    bias[:10] = fc2.bias.detach().numpy()

    sec.addConst(tmp_fc2, bias, inplace=True)
    
    return tmp_fc2