import numpy as np
from fase.core.heaan import he
from fase.core.commonAlgo import CommonAlgorithms

def linear(context, weight, bias, ctx, activation):
    """activation could be [ReLU, square, ..]
    """
    output = CommonAlgorithms.mat_mult_diag(context, weight, ctx)

    context.scheme.addAndEqual(output, bias)
    
    output = activation(context, output)
    return output


def batchnorm(context, ctx, gamma, beta, mu, sigma, eps=1e-8):
    """
    eps를 그냥 더하는 걸로 충분할까? 
    """
    frac = gamma / np.sqrt(sigma + eps)
    bias = -gamma * mu / np.sqrt(sigma + eps) + beta

    context.scheme.multByConst(ctx, frac, inplace=True)
    context.scheme.add(ctx, bias, inplace=True)

    return ctx



def conv2d(context, window, ctx, pad=0, stride=1):
    """

    """
    out = he.Ciphertext()
    # loop
    tmp = context.scheme.multByConst(ctx, window, inplace=True)
    #CommonAlgorithms.reduce(out)

    context.scheme.add(out, tmp, inplace=True)


def average_pool(context, ctx):
    """Average pooling as a strided convolution with uniform windows
    """
    window = [[1/4,1/4],[1/4,1/4]]
    output = conv2d(context, window, ctx, 0, 2)
    return output


#def ReLU