import numpy as np
import argparse
import fase
from time import time

parser = argparse.ArgumentParser()

parser.add_argument("--fpga", dest='use_fpga', action='store_true')
parser.add_argument("--cuda", dest='use_cuda', action='store_true')
args = parser.parse_args()

if args.use_fpga:
    fase.USE_FPGA = True
if args.use_cuda:
    fase.USE_CUDA = True

#from fase import HEAAN

from fase.core import heaan
he = heaan.he


def check_consistency(dec, val):
    good = np.all(np.isclose(dec[:len(val)], val, rtol=0.01))
    print("are values close to each other?:", good)
    if not good:
        print("v1", dec)
        print("v2", val)


t0 = time()

logp = 30
logq = logp + 120
logn = 7
n = nslots = 1 * 2 ** logn

ckks = heaan.HEAANContext(logn, logp, logq)
t1 = time()
print(f"Context ready {t1 - t0:.2f}")

print("... testing encryption") 
val = [1,2,3,4]
ctxt = ckks.encrypt(val)
dec = ckks.decrypt(ctxt)
check_consistency(dec, val)


print("\n ... testing addition") 
v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])
ctxt1 = ckks.encrypt(v1)
ctxt2 = ckks.encrypt(v2)
ctxt3 = ckks.add(ctxt1, ctxt2)
check_consistency(ckks.decrypt(ctxt3), v1+v2)

# inplace option
ckks.add(ctxt1, ctxt2, inplace=True)
print("in_place ver.")
check_consistency(ckks.decrypt(ctxt1), v1+v2)

t2 =time()
print("\n ... testing multiplication") 
v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])
ctxt1 = ckks.encrypt(v1)
ctxt2 = ckks.encrypt(v2)
ctxt3 = ckks.mult(ctxt1, ctxt2, inplace=False)
print(f"Mult took {time()-t2:.3f}")
check_consistency(ckks.decrypt(ctxt3), v1*v2)

# inplace option
ckks.mult(ctxt1, ctxt2, inplace=True)
print("in_place ver.")
check_consistency(ckks.decrypt(ctxt1), v1*v2)


# Bootstrap
dd = ckks.decrypt(ctxt)
t3 =time()
ckks.bootstrap_inplace(ctxt)
print(f"bootstrapping done in {time() - t3:.3f}")
check_consistency(ckks.decrypt(ctxt), dd)

print(f"Test done in {time()-t0:.2f}")