import numpy as np
from fase.core.commonAlgo import CommonAlgorithms


def batchnorm(scheme, ctx, m, c, h, gamma, beta, running_mu, running_sigma, eps=1e-8):
    mu = running_mu.reshape(1,c,1,1)
    sigma = running_sigma.reshape(1,1,1)
    
    scheme.sub(ctx, mu, inplace=True)
    scheme.multByConst(ctx, 1/np.sqrt(sigma + eps))
    #xhat = (x - mu)/np.sqrt(sigma + 1e-8)
    
    return gamma.reshape(1,c,1,1) * xhat + beta.reshape(1,c,1,1)



#def average_pooling(scheme, ctx, m, c, h):
    