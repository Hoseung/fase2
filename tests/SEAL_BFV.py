#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fase import seal


# 계산을 시작하기 전 scheme의 context를 설정해야함.  
# 설정에 필요한 parameter들은 `EncryptionParameters`로 설정가능.  
# 주요한 parameter는 다음과 같음 :
# - poly_modulus_degree
# - coeff_modulus ([ciphertext] coefficient modulus)
# - plain_modulus (plaintext modulus; only for the BFV scheme).

# In[2]:


parms = seal.EncryptionParameters(seal.scheme_type.bfv)


# ### Noise budget
# FHE scheme들은 inverse 계산을 불가능하게 하기 위해 계산에 noise를 집어넣음.  
# 그런데 연산이 진행되면 noise가 점점 증가함 (특히 ciphertext * ciphertext) 그래서 noise budget이라는 개념이 사용됨.  
# 문제의 Multilication depth를 미리 확인하고 적절한 noise budget을 설정해야함.
# 
# #### poly_modulus_degree
# Polynomical modulus의 degree를 뜻하며, 2^x 형태임. 일반적으로 2^10, ^11, .. ^15정도를 사용함. 

# In[3]:


p = 4096
parms.set_poly_modulus_degree(p)


# #### (Ciphertext) coefficient modulus
# 서로 다른 prime number의 곱인 어떤 수이며, 각각의 prime number는 최대 60bit임.  
# coeff_modulus가 클 수록 noise budget이 커짐.
# coeff_modulus의 최대치는 poly_modulus_degree로 결정 됨.  
# 
# 
#         +----------------------------------------------------+
#         | poly_modulus_degree | max coeff_modulus bit-length |
#         +---------------------+------------------------------+
#         | 1024                | 27                           |
#         | 2048                | 54                           |
#         | 4096                | 109                          |
#         | 8192                | 218                          |
#         | 16384               | 438                          |
#         | 32768               | 881                          |
#         +---------------------+------------------------------+
# 

# In[4]:


parms.set_coeff_modulus(seal.CoeffModulus.BFVDefault(p))

## 참고) poly_modulus_degree에 따른 max coeff_modulus 계산해주는 함수
seal.CoeffModulus.MaxBitCount(p)


# ### Plain_modulus
# plaintext modulus는 BFV에서 사용됨. 모든 정수가 가능하지만 prime number면 더 좋음.  
# plaintext modulus는 plaintext의 최대 크기를 결정하고, 거기에 따라 곱하기때 noise consumption이 결정 됨.  
# Plaintext를 가능한 작게 잡아야 좋음.  
# 
# noise budget = log2(coeff mod/plain mod) (bits)

# In[5]:


parms.set_plain_modulus(1024)


# (곱하기때) noise consumption = log2(plain_modulus) + (other terms) 이므로, 지금 상태론 곱하기 불가능! 

# In[6]:


context = seal.SEALContext(parms) # parameters are internally validated. 


# ### Key generation
# SEAL의 scheme들은 public key - sceret key 쌍으로 동작함. Secret key를 가진 사람만 원본 데이터를 볼 수 있음. (우리 과제는 한대의 컴퓨터에서 계산되므로 어느 키가 어디로 어떻게 가는지 신경 안 써도 됨.)

# In[7]:


keygen = seal.KeyGenerator(context)
secret_key = keygen.secret_key()
public_key = keygen.create_public_key()

# 근데 key를 만들때 쓰는 random number는 seed와 무관하겠지?


# ### Agents
# 
# encryption / decryption / ciphertext 계산을 위한 agent가 있음. 
# 
# encryptor는 public key를 사용하고,   
# decryptor는 secret key를 사용함.

# In[8]:


encryptor = seal.Encryptor(context, public_key)
evaluator = seal.Evaluator(context)
decryptor = seal.Decryptor(context, secret_key)


# ### Encoding to ptext
# 
# CKKS는 CKKSEncoder()가 따로 있으나, BFV에서는 integer를 hex로 바꾸어 넣어줌.

# In[9]:


def int_to_hex_string(vv):
    return f"{vv:x}"


# In[10]:


vv = 3
x_plain = seal.Plaintext(int_to_hex_string(vv))
print(x_plain.to_string()) # supposed to be the same as int_to_hex_string(vv)


# #### Encrypting to ctext

# In[11]:


def check_budget(ctxt):
    print(f"The size of encrypted text:, {ctxt.size()} bits") 
    print(f"Noise budget: {decryptor.invariant_noise_budget(ctxt)} bits")


# In[12]:


x_enc = encryptor.encrypt(x_plain)
check_budget(x_enc)
# 한참 남음. 


# In[13]:


x_dec = decryptor.decrypt(x_enc)
print(x_dec.to_string(), "==", int_to_hex_string(vv))
print("Seems to encrypt and decrypt correctly!")


# ### 곱하기 depth 줄이기 
# 계산을 짤때 곱하기 수를 줄이는게 좋음. x^4 + 2x^2 + 1은 곱하기가 최대 4번 되지만, (x^2+1) * (x^2+1)은 곱하기가 최대 3번임. 
# 
# 우리는 $$4x^4 + 8x^3 + 8x^2 + 8x + 4 = 4(x + 1)^2 \times (x^2 + 1)$$ 계산할것.

# In[14]:


print("Compute x^2+1")
x_sq = evaluator.square(x_enc)
x_sq_plus_one = evaluator.add_plain(x_sq, seal.Plaintext("1")) # 1은 hex로도 1

check_budget(x_sq_plus_one)


# budget이 좀 줄어들었음. 

# In[15]:


decrypted = decryptor.decrypt(x_sq_plus_one).to_string()
print(decrypted)


# In[16]:


# are they the same?
print(vv**2+1, int(decrypted, base=16))


# ***plain_modulus를 1024로 설정했기 때문에 결과값이 정답 % 1024로 표기됨. 

# In[17]:


# compute (x+1)^2
x_plus_1_sq = evaluator.add_plain(x_enc, seal.Plaintext('1'))
evaluator.square_inplace(x_plus_1_sq) # in_place 버전도 있음
check_budget(x_plus_1_sq)


# In[18]:


# multiply (x^2+1) * (x+1)^2 * 4
result = evaluator.multiply(x_sq_plus_one, x_plus_1_sq)
evaluator.multiply_plain_inplace(result, seal.Plaintext('4'))
check_budget(result)


# budget 거의 다 씀.

# In[19]:


dec_result = decryptor.decrypt(result)
print(4*vv**4+8*vv**3+8*vv**2+8*vv+4, int(dec_result.to_string(), base=16))
# 같음?


# ## Budget 관리: Relinearization
# 계산 중간중간 (곱하기 할때마다) relinearization을 해주면 ctxt 크기가 작아지고, 계산이 빨라지고, budget consumption이 줄어듬.   
# * Relinearization은 3bit polynomial을 2bit로 줄이는 것만 가능.
# * 'Relinearization key'가 따로 필요함. 한번에 하나씩
# * BFV와 CKKS에서 비슷하게 사용됨. 

# In[20]:


vv = 10
x_plain = seal.Plaintext(int_to_hex_string(vv))
x_enc = encryptor.encrypt(x_plain)


# In[21]:


relin_keys = keygen.create_relin_keys()

# 아까랑 같은데 군데군데 relin을 섞어줌.
x_sq = evaluator.square(x_enc)
evaluator.relinearize_inplace(x_sq, relin_keys)
x_sq_plus_one = evaluator.add_plain(x_sq, seal.Plaintext("1"))
check_budget(x_sq_plus_one)

x_plus_1_sq = evaluator.add_plain(x_enc, seal.Plaintext('1'))
evaluator.square_inplace(x_plus_1_sq)
evaluator.relinearize_inplace(x_plus_1_sq, relin_keys)

check_budget(x_plus_1_sq)

result = evaluator.multiply(x_sq_plus_one, x_plus_1_sq)
evaluator.relinearize_inplace(result, relin_keys)
evaluator.multiply_plain_inplace(result, seal.Plaintext('4'))
check_budget(result)


dec_result = decryptor.decrypt(result)
print(4*vv**4+8*vv**3+8*vv**2+8*vv+4, int(dec_result.to_string(), base=16))


# ## Batch encoding

# In[ ]:


from fase.core.seal_BFV import SEALBFVContext

sec = SEALBFVContext(poly_modulus_degree=p, prime_bit=20)

out = sec.encrypt([1,2,3,4,5, ])
print(sec.decrypt(out))

print("Everything works fine")

