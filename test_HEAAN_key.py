from fase import HEAAN
import fase.HEAAN as he
import numpy as np

class Param():
    def __init__(self, n=None, logn=None, logp=None, logq=None, logQboot=None):
        self.n = n
        self.logn = logn
        self.logp = logp
        self.logq = logq
        self.logQboot = logQboot
        if self.logn == None:
            self.logn = int(np.log2(n))


logq = 540
logp = 30
logn = 14
n = 1*2**logn
slots = n

parms = Param(n=n, logp=logp, logq=logq)

do_reduction=False
is_serialized = True

ring = he.Ring()
secretKey = he.SecretKey(ring)
scheme = he.Scheme(secretKey, ring, is_serialized, './test/')

algo = he.SchemeAlgo(scheme)

# reduction때는 right rotation N_class개 필요. 
if do_reduction:
    Nclass = Nmodel.head.shape[0]
    scheme.addLeftRotKeys(secretKey)
    for i in range(Nclass):
        scheme.addRightRotKey(secretKey, i+1) # 
else:
    # reduction 안 하면 하나짜리 rotation만 여러번 반복.
    scheme.addLeftRotKey(secretKey, 1)


