from numpy import polynomial as P
import math 

def approx_sign(n):
    """
    Approxiate sign function in [-1,1]
    """
    p_t1 = P.Polynomial([1,0,-1])
    p_x  = P.Polynomial([0,1])
    
    def c_(i: int):
        return 1/4**i * math.comb(2*i,i)

    def term_(i: int):
        return c_(i) * p_x * p_t1**i

    poly = term_(0)
    for nn in range(1,n+1):
        poly += term_(nn)
    return poly


def approx_relu(x, degree = 4, repeat=3):
    ff = approx_sign(degree)
    out = ff(x)
    for i in range(1,repeat):
        out = ff(out)

    return x * (out+1)/2
