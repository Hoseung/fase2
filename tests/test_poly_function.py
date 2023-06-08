import numpy as np
from fase import HEAAN as he

logq = 300
logp = 30
logn = 6
n = 1*2**logn
slots = n

ring = he.Ring()
secretKey = he.SecretKey(ring)
scheme = he.Scheme(secretKey, ring)

def decrypt(secretKey, enc):
    featurized = scheme.decrypt(secretKey, enc)
    arr = np.zeros(n, dtype=np.complex128)
    featurized.__getarr__(arr)
    return arr.real

algo = he.SchemeAlgo(scheme)

ctx = he.Ciphertext()
val = [0,1,2,3]
vv = np.zeros(n) # Need to initialize to zero or will cause "unbound"
vv[:len(val)] = val
scheme.encrypt(ctx, he.Double(vv), n, logp, logq)


res = he.Ciphertext()
degree = 4
algo.function_poly(res, ctx, he.Double(np.array([1,1,1,1])),logp, degree)

out = decrypt(secretKey, res)

x = np.array(val)

poly = lambda x : 1 + x + x**2 + x**3
plain_result = poly(x)

print(plain_result)
print("Close?", np.isclose(plain_result, out[:4], atol=1e-3))