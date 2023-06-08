import numpy as np
from typing import List
#from fase.core import heaan
from fase.core.heaan import HEAANContext 
#from fase.HEAAN import Ciphertext
from torch.nn import Conv2d
from torch import nn
from fase.nn.conv import get_out_size, gen_pad_mask, strided_indices

def rotate_for_conv(ctxt, 
                    nw:int, 
                    f_h:int, 
                    f_w:int, 
                    fhec:HEAANContext, 
                    stride:int =1)->List:
    """Rotate and pad with zero over/underflowed portion 
    
    Note
    ----
    Unlike SEAL where deeper level requires more slots (than necessary),
    HEAAN doesn't require more than the exact number of slots, 
    the side effect of which is that over/underflowed (= outside of convolution kernel) portion 
    of ctxt is non-zero. 
    So we zero out these slots manually.
    """
    rotated_ctxt = []
    hhw = int((f_w -1)/2)
    hhh = int((f_h -1)/2)
    for i in range(f_w)[::stride]:
        #print("rotate_for_conv", flush=True)
        for j in range(f_h)[::stride]:
            #print(i, j, (i-hhh)*nw + (j-hhw))
            if (i-hhh)*nw + (j-hhw) > 0:
                rotated_ctxt.append(fhec.lrot(ctxt, (i-hhh)*nw + (j-hhw)))
            elif (i-hhh)*nw + (j-hhw) < 0:
                rotated_ctxt.append(fhec.lrot(ctxt, fhec.parms.n + (i-hhh)*nw + (j-hhw)))
            elif (i-hhh)*nw + (j-hhw) == 0:
                rotated_ctxt.append(ctxt)
    return rotated_ctxt

# convolve_fhe
def convolve_fhe(rotated_ctxt:List, 
                kernel:np.ndarray, 
                fhec, 
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
                out_fhe = fhec.multByConst(rr, [rk]) # single value
                #print("11 out_fhe.logp, logq", out_fhe.logp, out_fhe.logq)
                fhec.rescale(out_fhe)
                if mask is not False:
                    mm = np.zeros(fhec.parms.n)
                    mm[:len(mask)] = mask
                    fhec.multByVec(out_fhe, mm, inplace=True) # x vector
                    #print("22 out_fhe.logp, logq", out_fhe.logp, out_fhe.logq)
                    fhec.rescale(out_fhe)
            else:
                tmp = fhec.multByConst(rr, [rk])
                fhec.rescale(tmp)
                if mask is not False:
                    mm = np.zeros(fhec.parms.n)
                    mm[:len(mask)] = mask
                    fhec.multByVec(tmp, mm, inplace=True)
                    fhec.rescale(tmp)
                    
                fhec.add(out_fhe, tmp, inplace=True)
                #print(fhec.decrypt(out_fhe), flush=True)

    return out_fhe

def my_conv2D_FHE(fhec:HEAANContext, 
                img_enc:List, 
                nh:int, 
                nw:int, 
                convlayer:Conv2d,
                stride_in:int =1, 
                stride_out:int =1, 
                padding:str ="same"):    
    """dd
    
    """
    kernel = convlayer.weight.detach().numpy()
    bias = convlayer.bias.detach().numpy()

    c_out, c_in, f_h, f_w = kernel.shape
    # e.g., new stride_2 to strided_2 = stride_4
    stride_out *= stride_in
    out_nh, out_nw = get_out_size((nh, nw), (f_h, f_w), 
                                  stride=stride_out, 
                                  padding=padding)

    print("Output image size", out_nh, out_nw, flush=True)
    #print("a", flush=True)
    # 3 - > 5, 5 - > 9 , ... 
    dilated_shape = ((f_h-1)*stride_in+1,
                     (f_w-1)*stride_in+1,)
    #dilated_kernel = np.zeros(dilated_shape)

    # Striding by using mask
    mask = np.zeros(fhec.parms.n)
    if stride_out == 1:
        mask[:nw*nh] = np.ones(nw*nh)
    else :
        _mask = np.zeros((nw,nh))
        _mask[::stride_out,::stride_out] = 1.
        mask[:nw*nh] = _mask.ravel()

    # padding mask
    pad_masks = gen_pad_mask((f_h, f_w), (nh, nw))

    # rotate each channel
    rotated =[]
    for channel_enc in img_enc:
        #print("running", flush=True)
        rotated.append(rotate_for_conv(channel_enc, 
                                   nw, 
                                   dilated_shape[0],
                                   dilated_shape[1], 
                                   fhec,
                                   stride=stride_in))

    #print("go", flush=True)
    ### Main loop ###
    conv_out=[]
    cnt = 0
    for this_kernel_in_channel, this_bias in zip(kernel, bias):
        #print(cnt, bias, flush=True)
        # convolve each channel of image and kernel
        result_each_out_channel = None
        for this_channel, this_kernel in zip(rotated, this_kernel_in_channel):
            if result_each_out_channel == None:
                result_each_out_channel = convolve_fhe(this_channel, this_kernel, fhec, pad_masks)
                #print(hec.decrypt(result_each_out_channel)
            else:
                fhec.add(result_each_out_channel,
                        convolve_fhe(this_channel, this_kernel, fhec, pad_masks), inplace=True)

        fhec.rescale(result_each_out_channel)
        c_tmp = fhec.encrypt(this_bias*mask, 
                            n = fhec.parms.n,
                            logp = result_each_out_channel.logp, logq = result_each_out_channel.logq)
        fhec.add(result_each_out_channel, c_tmp, inplace=True)

        conv_out.append(result_each_out_channel)  
        cnt += 1
    
    return conv_out, out_nh, out_nw

# def do_bootstrap(fhec, ctxt):
#     ctxt = fhec.decrypt(ctxt)
#     return fhec.encrypt(ctxt)

def do_bootstrap(fhec, ctxt):
    """
    """
    return fhec.bootstrap(ctxt, fhec.parms.logqBoot)

def approx_relu_fhe(fhec, calgo, ctxts, ff, xfactor = 20, repeat = 3):
    poly_mult_depth = 5
    output = []
    for ictx, ctx in enumerate(ctxts):

        scaled = fhec.multByConst(ctx, [1/xfactor], inplace=False)
        fhec.rescale(scaled)

        tmp = calgo.function_poly(ff.coef, scaled)

        for _ in range(1,repeat):
            if tmp.logq < (fhec.parms.logqBoot + poly_mult_depth * fhec.parms.logp):
                do_bootstrap(fhec, tmp)
                print("Bootstrapped", ictx, _)
            tmp = calgo.function_poly(ff.coef, tmp)
            print(fhec.decrypt(tmp))
            print(tmp)

        # (out + 1)
        fhec.addConst(tmp, [1], inplace=True)
        # (out + 1) / 2
        fhec.multByVec(tmp, np.repeat(0.5, 1024), inplace=True)
        fhec.rescale(tmp)

        #Mod mismatch between ctx and tmp
        if tmp.logp > ctx.logp:
            fhec.match_mod(tmp, ctx)
        elif tmp.logp < ctx.logp:
            fhec.match_mod(ctx, tmp)

        # x * (out + 1) /2
        fhec.mult(tmp, ctx, inplace=True)
        fhec.rescale(tmp)

        output.append(tmp)
    return output


def fhe_bn(fhec, ctxts, torch_bn, eps = 1e-5):
    """batch normalization of homomorphic CNN
    
    """
    gamma = torch_bn.weight.detach().cpu().numpy()
    beta = torch_bn.bias.detach().cpu().numpy()
    running_mean = torch_bn.running_mean.detach().cpu().numpy()
    running_var = torch_bn.running_var.detach().cpu().numpy()
    
    # Per channel
    result = []
    slot_count = fhec.parms.n
    for i, this_channel in enumerate(ctxts):
        denom = np.sqrt(running_var[i] + eps)
        factor = gamma[i]/denom
        const  = -running_mean[i]/denom*gamma[i] + beta[i]

        fhec.multByVec(this_channel, np.repeat(factor, slot_count), inplace=True)
        fhec.rescale(this_channel)
        fhec.addConst(this_channel, [const], inplace=True)
        result.append(this_channel)
        
    return result


def rotate_for_pool(ctxt, 
                    nw:int, 
                    f_h:int, 
                    f_w:int, 
                    fhec, 
                    stride:int =1)->List:
    """
    Tested only for 2x2, stride=2 pooling
    """

    rotated_ctxt = []
    for i in range(f_w)[::stride]:
        for j in range(f_h)[::stride]:
            rot = i*nw + j
            if rot > 0:
                rotated_ctxt.append(fhec.lrot(ctxt, rot))
            elif rot == 0:
                rotated_ctxt.append(ctxt)
            
    return rotated_ctxt


def fhe_avg_pool(fhec, ctxts, nh, nw, stride_in, kernel_size=2, padding = "same"):
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
    
    # rotate each channel
    rotated =[]
    for channel_enc in ctxts:
        #img_aug = aug_ctxt(channel_enc, nh, nw, sec)
        rotated.append(rotate_for_pool(channel_enc, nw, 
                                       dilated_shape[0],
                                       dilated_shape[1], 
                                       fhec, 
                                       stride=stride_in))


    # Striding uses mask
    mask = np.zeros(fhec.parms.n)

    _mask = np.zeros((nw,nh))
    _mask[::stride_out,::stride_out] = 1. # to match torch.Avgpool2D behavior
    mask[:nw*nh] = _mask.ravel()

    # padding mask
    output = []
    for this_channel in rotated:
        tmp = convolve_fhe(this_channel, kernel, fhec, None)
        fhec.multByVec(tmp, mask, inplace=True)
        fhec.rescale(tmp)
        output.append(tmp)
    return output, out_nh, out_nw


def reshape(fhec, ctxts):
    # tmp10
    nx = ny = 32

    ind_put = 0
    ind_source = strided_indices(nx, ny, 8, 8)
    for ich, tt in enumerate(ctxts):
        for ind_ch in ind_source:
            if ind_put == 0:
                mask = np.zeros(fhec.parms.n)
                mask[ind_ch] = 1.
                output = fhec.multByVec(tt, mask)
                fhec.rescale(output)
                #print(fhec.decrypt(output))
                #print(output.logp, output.logq)

            mask = np.zeros(fhec.parms.n)
            mask[ind_ch] = 1.
            masked = fhec.multByVec(tt, mask)
            fhec.rescale(masked)
            #print(fhec.decrypt(masked))
            #print(masked.logp, masked.logq)

            fhec.lrot(masked, ind_ch - ind_put, inplace=True)
            
            fhec.add(output, masked, inplace=True)
            #print(fhec.decrypt(output))
            #print(output.logp, output.logq)
            ind_put += 1

    return output

from fase.core.commonAlgo import CommonAlgorithms
def fullyConnected(fhec, ctxt, fc_layer: nn.modules.Linear):
    """Compute a FullyConnected layer
    
    parameters
    ----------
    ctxt: single Ciphertext carrying a n-dim feature in compact packing
    
    
    """
    calgo = CommonAlgorithms(fhec, "HEAAN")
    tmp_fc = fhec.encrypt(np.zeros(fhec.parms.n)) 
    # Do I have to encrypt zeros or can I always assume an empty ctxt on creating a new one?

    ind_put = 0
    for ww in fc_layer.weight.detach().numpy():
        q = np.zeros(fhec.parms.n)
        q[:len(ww)] = ww
        feature_weight = fhec.multByVec(ctxt, q, rescale=True)

        # Sum
        reduced = calgo._sum_reduce_he(feature_weight, fhec.parms.logn, fhec._scheme)

        mask = np.zeros(fhec.parms.n)
        mask[ind_put] = 1.
        masked = fhec.multByVec(reduced, mask, rescale=True)

        if ind_put == 0: fhec.match_mod(tmp_fc, masked)
        fhec.add(tmp_fc, masked, inplace=True)
        ind_put += 1

    bias = np.zeros(fhec.parms.n)
    bb = fc_layer.bias.detach().numpy()
    bias[:len(bb)] = bb

    c_tmp = fhec.encrypt(bias, 
                         n = fhec.parms.n,
                         logp = tmp_fc.logp, 
                         logq = tmp_fc.logq)
    fhec.add(tmp_fc, c_tmp, inplace=True)
    
    return tmp_fc