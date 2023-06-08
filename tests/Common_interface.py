from fase.core import common
import numpy as np

from fase import seal
from fase import HEAAN as he

from fase.core.common import HEAANContext, HEAANParameters, SEALContext

# SEAL setup
poly_modulus_degree = 16384
coeff_moduli = [37, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 37]

ckks_se = SEALContext(poly_modulus_degree=poly_modulus_degree,
                             coeff_moduli=coeff_moduli,
                             scale_bit=28)
# HEAAN setup
ckks_he = HEAANContext(2, 30, 150)


# ## encrypt and decrypt
for ckks in [ckks_he, ckks_se]:
    val = [1,2,3,4]
    ctxt = ckks.encrypt(val)
    dec = ckks.decrypt(ctxt)
    print(dec[:len(val)])
    print(ckks._name)


# ## Add
# 
# * ckks supports direct addition and subtraction between Ciphertext and Plaintext

# In[9]:

print("Addition")
for ckks in [ckks_he, ckks_se]:
    v1 = [1,2,3,4]
    v2 = [5,6,7,8]
    ctxt1 = ckks.encrypt(v1)
    ctxt2 = ckks.encrypt(v2)
    ctxt3 = ckks.add(ctxt1, ctxt2)
    print(ckks.decrypt(ctxt3)[:len(v1)])
    
    # inplace option
    ckks.add(ctxt1, ctxt2, inplace=True)
    print("inplace", ckks.decrypt(ctxt1)[:len(v1)])
    print(ckks._name)


# # Subtract 

# In[4]:

print("Subtraction")
for ckks in [ckks_he, ckks_se]:
    v1 = [1,2,3,4]
    v2 = [5,6,7,8]
    ctxt1 = ckks.encrypt(v1)
    ctxt2 = ckks.encrypt(v2)
    ctxt3 = ckks.sub(ctxt1, ctxt2)
    print(ckks.decrypt(ctxt3)[:len(v1)])
    
    # inplace option
    ckks.sub(ctxt1, ctxt2, inplace=True)
    print(ckks.decrypt(ctxt1)[:len(v1)])
    
    print(ckks._name)


# ## Multiply by const value
print("Multiply with Plaintext")
for ckks in [ckks_he, ckks_se]:
    v1 = [1,2,3,4]
    v2 = [5,6,7,8]
    ctxt1 = ckks.encrypt(v1)
    ctxt2 = ckks.multByConst(ctxt1, v2)
    print(ckks.decrypt(ctxt2)[:len(v1)])
    
    ckks.multByConst(ctxt1, v2, inplace=True)
    print(ckks.decrypt(ctxt1)[:len(v1)], "inplace=True")
    print(ckks._name)


# ## Multiply by Ciphertext

# In[10]:

print("Mult ctxts")
for ckks in [ckks_he, ckks_se]:
    v1 = [1,2,3,4]
    v2 = [5,6,7,8]
    ctxt1 = ckks.encrypt(v1)
    ctxt2 = ckks.encrypt(v2)
    ctxt3 = ckks.mult(ctxt1, ctxt2, inplace=False)
    print(ckks.decrypt(ctxt3)[:len(v1)])
    
    # inplace option
    ckks.mult(ctxt1, ctxt2, inplace=True)
    print(ckks.decrypt(ctxt1)[:len(v1)])
    
    print(ckks._name)


# ## Square

# In[3]:


for ckks in [ckks_he, ckks_se]:
    v1 = [1,2,3,4]
    ctxt1 = ckks.encrypt(v1)
    ctxt3 = ckks.square(ctxt1)
    print(ckks.decrypt(ctxt3)[:len(v1)])
    
    #inplace option
    ckks.square(ctxt1, inplace=True)
    print(ckks.decrypt(ctxt1)[:len(v1)])
    
    print(ckks._name)


print("Rotation")
for ckks in [ckks_he, ckks_se]:
    v1 = [1,2,3,4]
    ctxt1 = ckks.encrypt(v1)
    ckks.lrot(ctxt1, 4)
    print("Rotate by 4")
    print(ckks.decrypt(ctxt1)[:len(v1)])
    print(ckks._name)


# ## Rescale

# Different approach to checking scale 
print("Rescaling")
for ckks in [ckks_he, ckks_se]:
    v1 = [1,2,3,4]
    ctxt1 = ckks.encrypt(v1)

    if ckks._name == "HEAAN":
        print("initial scale", ctxt1.logp)
    elif ckks._name == "SEAL":
        print("Initial Scale", np.log2(ctxt1.scale()))
    
    ctxt3 = ckks.square(ctxt1)

    if ckks._name == "HEAAN":
        print("scale after suqare()", ctxt3.logp)
    elif ckks._name == "SEAL":
        ckks.relin(ctxt3, ckks_se.relin_keys) # relinearization
        print("Scale after square", np.log2(ctxt1.scale()))
    
    ckks.rescale(ctxt3)
    if ckks._name == "HEAAN":
        print("scale after rescaling()", ctxt3.logp)
    elif ckks._name == "SEAL":
        print("Scale after rescaling", np.log2(ctxt1.scale()))
    #elif ckks._name == "SEAL":
        
    #    ckks.square(ctxt1, inplace=True)
    #    ckks.relin(ctxt1, ckks_se.relin_keys) # relinearization
    #    print("Scale after square", np.log2(ctxt1.scale()))

        # Rescale
    #    ckks.rescale(ctxt1)
        
        
        #print("after manual fix", np.log2(ctxt1.scale()))
        #print('\n')


# ## Mod switch
# 
# Different approach to checking scale 
for ckks in [ckks_he, ckks_se]:
    v1 = [1,2,3,4]
    ctxt1 = ckks.encrypt(v1)
    new_ct = ckks.encrypt([3,4,5,6])

    if ckks._name == "HEAAN":
        print("initial scale and mod", ctxt1.logp, ctxt1.logq)
        ctxt1 = ckks.square(ctxt1)

        print("scale after suqare()", ctxt1.logp)
        ckks.rescale(ctxt1)
        print("scale and mod after rescaling()", ctxt1.logp, ctxt1.logq)
        print("new_ct's mod", new_ct.logq)
        
        ckks.match_mod(new_ct, ctxt1)
        print("new_ct's mod switched", new_ct.logq)
        
    elif ckks._name == "SEAL":
        print("Initial Scale", np.log2(ctxt1.scale()))
        ckks.square(ctxt1, inplace=True)
        ckks._evaluator.relinearize_inplace(ctxt1, ckks_se.relin_keys) # relinearization
        print("Scale after square", np.log2(ctxt1.scale()))

        # Rescale
        ckks.rescale(ctxt1)
        print("Scale after rescaling", np.log2(ctxt1.scale()))
        
        print("after manual fix", np.log2(ctxt1.scale()))
        print("modulus after rescaling", ckks.context.get_context_data(ctxt1.parms_id()).chain_index())
        
        print("new_ct's modulus index", ckks.context.get_context_data(new_ct.parms_id()).chain_index())
        
        ckks.match_mod(new_ct, ctxt1)
        print("new_ct's modulus index after mod switch", ckks.context.get_context_data(new_ct.parms_id()).chain_index())
        print('\n')
    
    
    ckks.add(new_ct, ctxt1, inplace=True)
    print(ckks.decrypt(new_ct)[:len(v1)])
    print("Correct result = [4,8,14,22]")
    print('\n')





