import numpy as np
import fase.core.heaan as he

#n = parms.n
#logp = parms.logp
#logq = parms.logq
def decrypt_print(ctx, secretKey, n=20, ):
    res1 = decrypt(secretKey, ctx)
    print(res1[:n])

    
def decrypt(secretKey, enc):
    temp = scheme.decrypt(secretKey, enc)
    arr = np.zeros(n, dtype=np.complex128)
    temp.__getarr__(arr)
    return arr.real

def encrypt(val):
    ctxt = he.Ciphertext()#logp, logq, n)
    vv = np.zeros(n) # Need to initialize to zero or will cause "unbound"
    vv[:len(val)] = val
    scheme.encrypt(ctxt, he.Double(vv), n, logp, logq)
    del vv
    return ctxt

def sum_reduce(ctx, logn, scheme):
    """
    return sum of a Ciphertext (repeated nslot times)
    
    example
    -------
    sum_reduct([1,2,3,4,5])
    >> [15,15,15,15,15]
    """
    temp = he.Ciphertext()
    output = he.Ciphertext()
    
    for i in range(logn):
        if i == 0:
            scheme.leftRotateFast(temp, ctx, 2**i)
            scheme.add(output, ctx, temp)
        else:
            scheme.leftRotateFast(temp, output, 2**i)
            scheme.addAndEqual(output, temp)
            
    return output
