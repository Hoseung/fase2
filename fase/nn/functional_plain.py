import numpy as np

def avg_pooling_torch(prev_layer, filter_size=2, stride = 2):
    """
    Aveage pooling in pure Python. 
    input matrix (4-D array) expected to obey Pytorch tensor ordering (N, C, H, W)
    """

    (m, channels, n_H_prev, n_W_prev) = prev_layer.shape

    n_H = int((n_H_prev - filter_size)/filter_size + 1)
    n_W = int((n_W_prev - filter_size)/filter_size + 1)

    pooling = np.zeros((m,channels,n_H,n_W))

    for i in range(m):
        for c in range(channels):
            for h in range(n_H):
                for w in range(n_W):
                    prev_slice = prev_layer[i,
                                            c,
                                            h*stride : h*stride+filter_size,
                                            w*stride : w*stride+filter_size]
                    pooling[i,c,h,w] = np.mean(prev_slice)

    caches = (pooling,prev_layer,filter_size)                    

    return pooling, caches


def conv2d_eval_torch(x, weight, b, pad=2, stride=1):
    """
    2D convolution in pure Python. 
    input matrix (4-D array) expected to obey Pytorch tensor ordering (N, C, H, W)
    """
    (m, n_C_prev, n_h, n_w) = x.shape
    (n_C, n_C_prev, f,f) = weight.shape # f = kernel_size

    n_H = int(1 + (n_h + 2 * pad - f) / stride)
    n_W = int(1 + (n_w + 2 * pad - f) / stride)

    #print("Conv2D: output image size", (n_H, n_W))

    x_prev_pad = np.pad(x, ((0,0),(0,0), (pad,pad),(pad,pad)), 'constant', constant_values=0)
    #print(x_prev_pad.shape)

    Z = np.zeros((m, n_C, n_H,n_W)) 

    caches = (x,weight,b,pad,stride)

    for i in range(m):
        for c in range(n_C):
            for h in range(n_H):
                for w in range(n_W):
                    x_slice = x_prev_pad[i, 
                                         :,# 모든 채널 (3)
                                         h*stride:h*stride+f,
                                         w*stride:w*stride+f] 
                    Z[i,c,h,w] = np.sum(np.multiply(x_slice, weight[c,:,:,:]))

    return Z + b[None,:,None,None], caches

def fully_connected(prev_layer, w,b):
    """
    Fully connected layer.
    input matrix (4-D array) expected to obey Pytorch tensor ordering (N, C, H, W)
    """
    fc = prev_layer.dot(w)
    fc = fc+ b
    caches = {'input':prev_layer,'weights':w,'bias':b}
    return fc, caches

def approx_relu(x):
    return 0.47 + 0.5*x + 0.09*x**2


def batchnorm_eval_torch(x, gamma, beta, running_mu,running_sigma):
    """
    batch normalization layer.
    input matrix (4-D array) expected to obey Pytorch tensor ordering (N, C, H, W)
    """
    m,c,h,w = x.shape
    nt = (m*h*w)

    mu = running_mu.reshape(1,c,1,1)
    sigma = running_sigma.reshape(1,c,1,1)
    xhat = (x - mu)/np.sqrt(sigma + 1e-8)

    return gamma.reshape(1,c,1,1) * xhat + beta.reshape(1,c,1,1)