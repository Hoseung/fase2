import numpy as np
import pickle
from typing import List#, Callable

from fase.core.heaan import he
#from .fhe_utils import sum_reduce#, encrypt
from fase.core import commonAlgo

class HETreeFeaturizer:
    """Featurizer used by the client to encode and encrypt data.
       모든 Context 정보를 다 필요로 함. 이것만 따로 class를 만들고 CKKS context 보내기 좀 귀찮은데? 
    """
    def __init__(self, comparator: np.ndarray,
                 scheme, 
                 ckks_parms,
                 use_symmetric_key=False):
        self.comparator = comparator
        self.scheme = scheme
        #self.encoder = encoder
        self._parms = ckks_parms
        self.use_symmetric_key = use_symmetric_key
        

    def encrypt(self, x: np.ndarray):
        features = x[self.comparator]
        features[self.comparator == -1] = 0
        features = list(features)

        ctx = self._encrypt(features)
        return ctx

    def _encrypt(self, val, n=None, logp=None, logq=None):
        if n == None: n = self._parms.n
        if logp == None: logp = self._parms.logp
        if logq == None: logq = self._parms.logq

        ctxt = he.Ciphertext()#logp, logq, n)
        vv = np.zeros(n) # Need to initialize to zero or will cause "unbound"
        vv[:len(val)] = val
        self.scheme.encrypt(ctxt, he.Double(vv), n, logp, logq)
        del vv
        return ctxt

    def save(self, path:str):
        pickle.dump(self.comparator, open(path, "wb"))

from fase.core import commonAlgo
class HETreeEvaluator:
    """Evaluator which will perform homomorphic computation"""

    def __init__(self, 
                 model,
                 #b0: np.ndarray, w1, b1, w2, b2,
                 scheme,
                 parms,
                 activation_coeffs: List[float], 
                 sk=None,
                 #polynomial_evaluator: Callable,
                 
                 #relin_keys: seal.RelinKeys, galois_keys: seal.GaloisKeys, scale: float,
                 do_reduction=True,
                 silent=False):
        """Initializes with the weights used during computation.

        Args:
            b0: bias of the comparison step

        """        
        b0, w1, b1, w2, b2 = model.return_weights()

        self.sk = sk
        self.scheme = scheme
        self.algo = he.SchemeAlgo(scheme)
        self.commonAlgo = commonAlgo.CommonAlgorithms(scheme, "HEAAN")
        # scheme should hold all keys
        self.parms = parms
        
        self._activation_coeff = activation_coeffs
        self._activation_poly_degree = len(activation_coeffs) -1
        self.do_reduction = do_reduction

        # 10-degree activation -> up to 5 multiplications 
        logq_w1 = self.parms.logq - 5 * self.parms.logp
        logq_b1 = logq_w1 - self.parms.logp
        logq_b2 = logq_b1 - 5*self.parms.logp

        self.b0_ctx = self.encrypt(b0)
        #self.b0 = b0
        self.w1 = [self.to_double(w) for w in w1]
        #self.b1 = b1
        self.w2 = [self.to_double(w) for w in w2]
        self.b1_ctx = self.encrypt(b1, logq=logq_b1)
        self.b2_ctx = [self.encrypt(b, logq=logq_b2) for b in b2]

        if not silent: self.setup_summary()      
    
    def setup_summary(self):
        print("CKKS paramters:")
        print("---------------------------")
        print(f"n = {self.parms.n}")
        print(f"logp = {self.parms.logp}")
        print(f"logq = {self.parms.logq}")
        print(f"tanh activation polynomial coeffs = {self._activation_coeff}")
        print(f"tanh activation polynomial degree = {self._activation_poly_degree}")
        
        print("\nNeural RF")
        print("---------------------------")
        print(f"")
    
    def heaan_double(self, val):
        mvec = np.zeros(self.parms.n)
        mvec[:len(val)] = np.array(val)
        return he.Double(mvec)

    def decrypt_print(self, ctx, n=20):
        res1 = self.decrypt(ctx)
        print("_____________________")
        print(res1[:n])
        print(res1.min(), res1.max())
        print("---------------------")

    def decrypt(self, enc):
        temp = self.scheme.decrypt(self.sk, enc)
        arr = np.zeros(self.parms.n, dtype=np.complex128)
        temp.__getarr__(arr)
        return arr.real
        
    def encrypt_ravel(self, val, **kwargs):
        """encrypt a list
        """
        return self.encrypt(np.array(val).ravel(), **kwargs)

    def encrypt(self, val, n=None, logp=None, logq=None):
        if n == None: n = self.parms.n
        if logp == None: logp = self.parms.logp
        if logq == None: logq = self.parms.logq
            
        ctxt = he.Ciphertext()
        vv = np.zeros(n) # Need to initialize to zero or will cause "unbound"
        vv[:len(val)] = val
        self.scheme.encrypt(ctxt, he.Double(vv), n, logp, logq)
        del vv
        return ctxt
    
    def to_double(self, val):
        n = self.parms.n
        vv = np.zeros(n) # Need to initialize to zero or will cause "unbound"
        vv[:len(val)] = val
        return he.Double(vv)
        
        
    def activation(self, ctx):
        output = he.Ciphertext()
        #output = self.commonAlgo.function_poly(ctx, 
        #               he.Double(self._activation_coeff))
        output = he.Ciphertext()
        self.algo.function_poly(output, 
                    ctx, 
                    he.Double(self._activation_coeff), 
                    self.parms.logp, 
                    self._activation_poly_degree)
        return output        
        

    def __call__(self, ctx):
        # First we add the first bias to do the comparisons
        ctx = self.compare(ctx)
        #print("After compare")
        #self.decrypt_print(ctx)
        ctx = self.match(ctx)
        #print("after match")
        #self.decrypt_print(ctx)
        outputs = self.decide(ctx)
        if self.do_reduction:
            outputs = self.reduce(outputs)

        return outputs

    def compare(self, ctx, debug=False):
        """Calculate first layer of the HNRF
        
        ctx = featurizer.encrypt(x)
        
        Assuming n, logp, logq are globally available
        
        """
        b0_ctx = self.b0_ctx
        self.scheme.addAndEqual(ctx, b0_ctx)
        # Activation
        output = self.activation(ctx)
            
        del b0_ctx, ctx

        return output
    
    def _mat_mult(self, diagonals, ctx):
        """
        Take plain vector 
        """
        scheme = self.scheme
        n = self.parms.n
        logp = self.parms.logp
        #logq = self.parms.logq

        ctx_copy = he.Ciphertext()
        ctx_copy.copy(ctx)
        
        for i, diagonal in enumerate(diagonals):
            #print("logq in mat_mult", diagonal.logq, ctx_copy.logq)
            #scheme.modDownToAndEqual(diagonal, ctx.logq)
            if i > 0: scheme.leftRotateFastAndEqual(ctx_copy, 1) # r = 1

            # Multiply with diagonal
            dd = he.Ciphertext()
            #print("diagonal")
            #self.decrypt_print(diagonal,10)
            #print("ctx")
            #self.decrypt_print(ctx_copy,10)
            #scheme.mult(dd, diagonal, ctx_copy)
            
            # Reduce the scale of diagonal
            scheme.multByConstVec(dd, ctx_copy, diagonal, logp)
            scheme.reScaleByAndEqual(dd, logp)
            #print('dd')
            #print(dd)
            
            
            if i == 0:
                mvec = np.zeros(n)
                temp = he.Ciphertext()
                scheme.encrypt(temp, he.Double(mvec), n, logp, ctx_copy.logq - logp)
                ##scheme.modDownToAndEqual(temp, ctx_copy.logq)
                #print("temp",i)
                #print(temp)
                #self.decrypt_print(temp,10)
            
            # match scale 
            scheme.addAndEqual(temp, dd)

            #print("temp",i)
            #self.decrypt_print(temp,10)
            
            del dd
        del ctx_copy
        return temp


    def match(self, ctx):
        """Applies matching homomorphically.

        First it does the matrix multiplication with diagonals, then activate it.
        """
        output = self._mat_mult(self.w1, ctx)

        #print(f"MATCH:: 'output.logq', {output.logq} == {self.b1_ctx.logq}?")
        self.scheme.addAndEqual(output, self.b1_ctx)
        
        output = self.activation(output)
        return output

    def decide(self, ctx):
        """Applies the decisions homomorphically.

        For each class, multiply the ciphertext with the corresponding weight of that class and
        add the bias afterwards.
        """
        # ww와 bb도 미리 modDowntoAndEqual 가능 
        outputs = []

        for ww, bb in zip(self.w2, self.b2_ctx):
            output = he.Ciphertext()
            
            # Multiply weights            
            #self.scheme.mult(output, ww, ctx)
            #print("\n output before", output)
            self.scheme.multByConstVec(output, ctx, ww, ctx.logp)
            #print("\n output after")
            #self.decrypt_print(output)
            #print("ctx and bb should be same in scale")
            #print("ctx", ctx)
            #print("bb", bb)
            self.scheme.reScaleByAndEqual(output, ctx.logp)
            
            # Add bias
            self.scheme.addAndEqual(output, bb)
            
            outputs.append(output)
        return outputs

    def _sum_reduce(self, ctx, logn, scheme):
        """
        return sum of a Ciphertext (repeated nslot times)
        
        example
        -------
        sum_reduct([1,2,3,4,5])
        >> [15,15,15,15,15]
        """
        output = he.Ciphertext()
        
        for i in range(logn):
            
            if i == 0:
                temp = he.Ciphertext(ctx.logp, ctx.logq, ctx.n)
                #print(i, ctx, temp)
                #print("reduce: ctx before rot")
                # self.decrypt_print(ctx,10)
                
                scheme.leftRotateFast(temp, ctx, 2**i)
                #print(i, ctx, temp)
                #print("reduce: before add")
                # self.decrypt_print(temp,10)
                scheme.add(output, ctx, temp)
                #print("reduce: after add")
                # self.decrypt_print(output,10)
            else:
                scheme.leftRotateFast(temp, output, 2**i)
                #print(i, output, temp)
                #print("reduce: before add")
                # self.decrypt_print(output,10)
                # self.decrypt_print(temp,10)
                scheme.addAndEqual(output, temp)
                #print("reduce: after add")
                # self.decrypt_print(output,10)
        return output


    def reduce(self, outputs):
        logp = self.parms.logp
        scheme = self.scheme

        for i, output in enumerate(outputs):
            # print("reduce before",)
            # self.decrypt_print(output,10)
            #output = sum_reduce(output, self.parms.logn, self.scheme)
            output = self._sum_reduce(output, self.parms.logn, self.scheme)

            # print("reduce after",)
            # self.decrypt_print(output,10)

            mask = np.zeros(self.parms.n)
            mask[0] = 1
            mask_hedb = he.ComplexDouble(mask)
            if i == 0:
                scores = he.Ciphertext()
                scheme.multByConstVec(scores, output, mask_hedb, logp)
                # print("reduce score",i)
                # self.decrypt_print(scores,10)
                # print("before rescale", scores)
                scheme.reScaleByAndEqual(scores, logp)
                # print("before rescale", scores)
            else:
                temp = he.Ciphertext()
                scheme.multByConstVec(temp, output, mask_hedb, logp)
                # print("reduce score",i)
                # self.decrypt_print(scores,10)
                # print("before rescale", scores)
                scheme.reScaleByAndEqual(temp, logp)
                # print("after rescale", scores)
                scheme.rightRotateFastAndEqual(temp, i)
                scheme.addAndEqual(scores, temp)

        return scores


    @classmethod
    def from_model(cls, model,
                   scheme,
                   parms,
                   activation_coeffs: List[float],
                   sk=None,
                   do_reduction=False):
        """Creates an Homomorphic Tree Evaluator from a model, i.e a neural tree or
        a neural random forest. """
        #b0, w1, b1, w2, b2 = model.return_weights()

        return cls(model, scheme, parms, activation_coeffs, sk=sk, do_reduction=do_reduction)
