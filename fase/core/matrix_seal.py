import numpy as np
from fase import seal

# https://eprint.iacr.org/2018/1041.pdf 

def sigma_diagonal_vector(d: int, k:int) -> np.array:
    """Creates the k-th diagonal for the sigma operator
    for matrices of dimension dxd."""
    
    u = np.arange(d**2)
    if k >= 0:
        index = (u - d*k >= 0) & (u < d*k + d - k)
    else:
        index = (u - d*(d+k) >= -k ) & (u - d*(d+k)< d)
    u[index] = 1
    u[~index] = 0
    return u

def tau_diagonal_vector(d: int, k:int) -> np.array:
    """Creates the k-th diagonal for the tau operator
    for matrices of dimension dxd."""
    
    u = np.zeros(d**2)
    for i in range(d):
        l = (k + d * i)
        u[l] = 1
    return u

def row_diagonal_vector(d,k):
    v_k = np.arange(d**2)
    index = (v_k % d) < (d - k)
    v_k[index] = 1
    v_k[~index] = 0
    
    v_k_d = np.arange(d**2)
    index = ((v_k_d % d) >= (d -k)) & ((v_k_d % d) < d)
    v_k_d[index] = 1
    v_k_d[~index] = 0
    return v_k, v_k_d

def column_diagonal_vector(d,k):
    v_k = np.ones(d**2)
    return v_k

class MatrixMultiplicator:
    """Base class to create a matrix multiplicator operator."""
    def __init__(self, d, create_zero, sigma_diagonal_vector, tau_diagonal_vector,
                 row_diagonal_vector, column_diagonal_vector,
                 rotate=None, add=None, pmult=None, cmult=None):
        
        self.d = d
        self.create_zero = create_zero
        self.sigma_diagonal_vector = sigma_diagonal_vector
        self.tau_diagonal_vector = tau_diagonal_vector
        self.row_diagonal_vector = row_diagonal_vector
        self.column_diagonal_vector = column_diagonal_vector
        
        if not rotate:
            rotate = lambda x,k: np.roll(x, -k)
        if not add:
            add = lambda x,y: x+y
        if not pmult:
            pmult = lambda x,y: x*y
        if not cmult:
            cmult = lambda x,y: x*y
            
        self.rotate, self.add, self.pmult, self.cmult = rotate, add, pmult, cmult
    
    def sigma_lin_transform(self, input):
        
        sigma = []
        d = self.d
    
        for k in range(-d+1,d):
            sigma.append(self.sigma_diagonal_vector(d,k))
        
        output = self.create_zero()
        
        for sigma_vector,k in zip(sigma,range(-d+1,d)):
            output = self.add(output, self.pmult(self.rotate(input,k), sigma_vector))
        return output
    
    def tau_lin_transform(self, input):

        tau = []
        d = self.d

        for k in range(d):
            tau.append(self.tau_diagonal_vector(d,k))
            
        output = self.create_zero()
        
        for tau_vector,k in zip(tau,range(d)):
            output = self.add(output, self.pmult(self.rotate(input,k * d), tau_vector))
        return output
    
    def row_lin_transform(self, input, k):
        
        d = self.d
        v_k, v_k_d = self.row_diagonal_vector(d, k)
        
        output = self.create_zero()
        
        output = self.add(output, self.pmult(self.rotate(input, k), v_k))
        output = self.add(output, self.pmult(self.rotate(input, k-d), v_k_d))

        return output
    
    def column_lin_transform(self, input, k):
        
        d = self.d
        v_k = self.column_diagonal_vector(d, k)
        
        output = self.create_zero()
        
        output = self.add(output, self.pmult(self.rotate(input, d*k),v_k))

        return output
    
    def matmul(self, A, B):
        
        d = self.d

        sigma_A = self.create_zero()
        sigma_A = self.sigma_lin_transform(A)

        tau_B = self.create_zero()
        tau_B = self.tau_lin_transform(B)

        output = self.cmult(sigma_A, tau_B)

        for k in range(1,d):
            shift_A = self.row_lin_transform(sigma_A, k)
            shift_B = self.column_lin_transform(tau_B, k)

            output = self.add(output, self.cmult(shift_A, shift_B))
        
        return output
        

def encode_matrices_to_vector(matrix):
    shape = matrix.shape
    assert len(shape) == 3, "Non tridimensional tensor"
    assert shape[1] == shape[2], "Non square matrices"
    
    g = shape[0]
    d = shape[1]
    n = g * (d ** 2)
    
    output = np.zeros(n)
    for l in range(n):
        k = l % g
        i = (l // g) // d
        j = (l // g) % d
        output[l] = matrix[k,i,j]
        
    return output

def decode_vector_to_matrices(vector, d):
    n = len(vector)
    g = n // (d ** 2)
    
    output = np.zeros((g, d, d))
    
    for k in range(g):
        for i in range(d):
            for j in range(d):
                output[k,i,j] = vector[g * (d*i + j) +k]
    return output

def encode_matrix_to_vector(matrix: np.array) -> np.array:
    """Encodes a d*d matrix to a vector of size d*d"""
    shape = matrix.shape
    assert len(shape) == 2 and shape[0] == shape[1], "Non square matrix"
    d = shape[0]
    output = np.zeros(d**2)
    for l in range(d**2):
        i = l // d
        j = l % d
        output[l] = matrix[i,j]
    return output

def decode_vector_to_matrix(vector):
    n = len(vector)
    d = np.sqrt(n)
    assert len(vector.shape) == 1 and d.is_integer(), "Non square matrix"
    d = int(d)
    
    output = np.zeros((d,d))
    
    for i in range(d):
        for j in range(d):
            output[i,j] = vector[d*i + j]
    return output

def weave(vector, g):
    output = np.zeros(len(vector) * g)
    for i in range(len(vector)):
        output[i*g:(i+1)*g] = vector[i]
    return output




######### Ciphertext ##########
def sum_reduce(ctx: seal.Ciphertext, evaluator: seal.Evaluator, 
               galois_keys: seal.GaloisKeys, n_slot: int):
    """Sums all the coefficients of the ciphertext, supposing that coefficients up to n_slot 
    are non zero. The first coefficient of the output will then be the sum of the coefficients."""
    n = int(np.ceil(np.log2(n_slot)))
    
    temp = seal.Ciphertext()
    output = seal.Ciphertext()
    
    for i in range(n):
        if i == 0:
            evaluator.rotate_vector(ctx, 2**i, galois_keys, temp)
            evaluator.add(ctx, temp, output)
        else:
            evaluator.rotate_vector(output, 2**i, galois_keys, temp)
            evaluator.add_inplace(output, temp)
    return output

class Seal_matmult():
    """SEAL implementation of Jiang+18 matrix multiplication
    """
    def __init__(self, scheme):
        self.scheme = scheme
        self.encoder = scheme._encoder
        self.encryptor = scheme._encryptor
        self.evaluator = scheme._evaluator
        self.scale = scheme.scale
        try:
            self.decryptor = self.scheme._decryptor
        except:
            print("Not a SecretKey holder")


    def gen_multiplicator(self, d,k):
        ckks_sigma_diagonal_vector = lambda d,k: self.encode(sigma_diagonal_vector(d,k))
        ckks_tau_diagonal_vector = lambda d,k: self.encode(tau_diagonal_vector(d,k))
        ckks_row_diagonal_vector = lambda d,k: [self.encode(vector) for vector in row_diagonal_vector(d,k)]
        ckks_column_diagonal_vector = lambda d,k: self.encode(column_diagonal_vector(d,k))

        self.cmm = MatrixMultiplicator(d, self.ckks_create_zero, ckks_sigma_diagonal_vector, ckks_tau_diagonal_vector,
                                ckks_row_diagonal_vector, ckks_column_diagonal_vector, self.ckks_rotate, self.ckks_add, 
                                self.ckks_pmult, self.ckks_cmult)

    def get_vector(self, ctx):
        ptx = seal.Plaintext()
        self.decryptor.decrypt(ctx, ptx)
        return np.array(self.encoder.decode(ptx))

    def encode(self, vector):
        ptx = seal.Plaintext()    
        return self.encoder.encode(vector, self.scale)

    def encrypt(self, vector):
        ptx = self.encode(vector)
        ctx = seal.Ciphertext()
        ctx = self.encryptor.encrypt(ptx)
        return ctx

    def ckks_create_zero(self):
        zero = np.zeros(self.encoder.slot_count())
        ptx = seal.Plaintext()
        ptx = self.encoder.encode(zero, self.scale)
        ctx = self.encryptor.encrypt(ptx)#seal.Ciphertext()
        return ctx

    def ckks_rotate(self, ctx, k):
        output = seal.Ciphertext()
        output = self.evaluator.rotate_vector(ctx, k, self.scheme.galois_keys)
        return output

    def ckks_add(self, ctx1, ctx2):
        output = seal.Ciphertext()
        if not ctx1.parms_id() == ctx2.parms_id():
            self.evaluator.mod_switch_to_inplace(ctx1, ctx2.parms_id())
        output = self.evaluator.add(ctx1, ctx2)
        return output

    def ckks_pmult(self, ctx, ptx):
        output = seal.Ciphertext()
        if not ptx.parms_id() == ctx.parms_id():
            self.evaluator.mod_switch_to_inplace(ptx, ctx.parms_id())
        output = self.evaluator.multiply_plain(ctx, ptx)
        self.evaluator.rescale_to_next_inplace(output)
        output.scale(self.scale)
        return output

    def ckks_cmult(self, ctx1, ctx2):
        output = seal.Ciphertext()
        if not ctx2.parms_id() == ctx1.parms_id():
            self.evaluator.mod_switch_to_inplace(ctx2, ctx1.parms_id())
        output = self.evaluator.multiply(ctx1, ctx2)
        self.evaluator.rescale_to_next_inplace(output)
        output.scale(self.scale)
        return output

    def __call__(self, A,B):
        ptx = seal.Plaintext()
        ptx = self.encoder.encode(A, self.scale)
        
        ctA = seal.Ciphertext()
        ctA = self.encryptor.encrypt(ptx)

        ptx = self.encoder.encode(B, self.scale)
        ctB = seal.Ciphertext()
        ctB = self.encryptor.encrypt(ptx)

        return self.cmm.matmul(ctA, ctB)
