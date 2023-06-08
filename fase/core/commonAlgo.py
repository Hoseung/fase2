from fase.core.heaan import he
# from fase import seal
from typing import List, Union
import numpy as np

# def coeffs_to_plaintext(coeffs: List[float], 
#                         encoder: seal.CKKSEncoder, 
#                         scale: float,
#                         broadcast=True) -> List[seal.Plaintext]:
#     """Computes the plaintext encodings of coefficients (SEAL-specific)
    
#     Called by the seal_polyeval function.

#     parameters
#     ----------
#     encoder: seal CKKS Encoder
#     scale: Scale of ctxt 
#     broadcast: if True, coefficients are multiplied to each of all slots. Defaults to True.

#     """
#     plain_coeffs = []
#     if broadcast:
#         repeat = encoder.slot_count()
#     else:
#         repeat = 1
#     for coef in coeffs:
#         plain_coeff = seal.Plaintext()
#         plain_coeff = encoder.encode([coef]*repeat, scale)
#         plain_coeffs.append(plain_coeff)
        
#     return plain_coeffs

# def compute_all_powers(ctx : seal.Ciphertext, 
#                        degree: int, 
#                        evaluator: seal.Evaluator, 
#                        relin_keys: seal.RelinKeys, 
#                        verbose=False) -> List[seal.Ciphertext]:
#     """Computes all powers of a given ciphertext (SEAL-specific)

#     Called by the seal_polyeval function.
    
#     parameters
#     ----------
#     ctx: seal.Ciphertext
#     degree: The highest degree of the polynomial
#     evaluator: SEAL evaluator of the current context
#     relin_keys: Keys used to relinearize multiplied ctxts
#     verbose: verbose
#     """
#     powers = [None] * (degree+1)
#     levels = np.zeros(degree+1)

#     powers[1] = ctx
#     levels[0] = levels[1] = 0
    
#     for i in range(2,degree+1):
            
#         minlevel = i
#         cand = -1
        
#         for j in range(1, i // 2 +1):
#             k = i - j
#             newlevel = max(levels[k],levels[j]) + 1
#             if newlevel < minlevel:
#                 cand = j
#                 minlevel = newlevel
                
#         if verbose:
#             print(f"i = {i}, i-cand = {i-cand}")
#             print(f"level for cand : {levels[cand]}, level for {i-cand} : {levels[i-cand]}")
#             print(f"minlevel = {minlevel}")
#             print(f"cand = {cand}")
        
#         levels[i] = minlevel
        
#         temp = seal.Ciphertext()
        
#         power_cand = powers[cand]
#         temp = evaluator.mod_switch_to(power_cand, powers[i-cand].parms_id())
#         temp = evaluator.multiply(temp, powers[i-cand])
#         evaluator.relinearize_inplace(temp, relin_keys)
#         evaluator.rescale_to_next_inplace(temp)
        
#         powers[i] = temp
        
#     return powers

# def multiply_and_add_coeffs(powers: List[seal.Ciphertext], plain_coeffs: List[seal.Plaintext],
#                             coeffs: List[float],
#                             evaluator: seal.Evaluator,
#                             scale: float,
#                             tol=1e-6) -> Union[seal.Ciphertext, List[seal.Ciphertext]]:
#     """Multiplies the coefficients with the corresponding powers andd adds everything.
    
#     If the polynomial is non-constant, returns the ciphertext of the polynomial evaluation.
#     Else if the polynomials is constant, the plaintext of the constant term is returned.

#     Called by the seal_polyeval function.

#     parameters
#     ----------
#     powers : list of power terms of cipherteKeys used to relinearize multiplied ctxtsxts such as [x, x^2, x^3, ...]
#     plain_coeffs: 
#     coeffs : coefficients of powers



#     """
#     assert len(powers) == len(plain_coeffs), f"Mismatch between the length of powers {len(powers)} and the length of coeffs {len(coeffs)}"
    
    
#     output = seal.Ciphertext()
#     a0 = plain_coeffs[0]
#     a0_added = False
    
#     temp = seal.Ciphertext()
    
#     for i in range(1, len(plain_coeffs)):
#         # We first check if the coefficient is not too small otherwise we skip it
#         coef = coeffs[i]
#         if np.abs(coef) < tol:
#             continue
            
#         plain_coeff = plain_coeffs[i]
#         power = powers[i]
        
#         evaluator.mod_switch_to_inplace(plain_coeff, power.parms_id())
        
#         temp = evaluator.multiply_plain(power, plain_coeff)
#         evaluator.rescale_to_next_inplace(temp)
        
#         if not a0_added:
#             evaluator.mod_switch_to_inplace(a0, temp.parms_id())
            
#             temp.scale(scale)
#             output = evaluator.add_plain(temp, a0)
#             a0_added = True
#         else:
#             evaluator.mod_switch_to_inplace(output, temp.parms_id())
#             # We rescale both to the same scale
#             output.scale(scale)
#             temp.scale(scale)
#             evaluator.add_inplace(output, temp)
#     if a0_added:
#         return output
#     else:
#         return a0

# def seal_polyeval(ctx : seal.Ciphertext, coeffs: List[float], 
#                   evaluator: seal.Evaluator, encoder : seal.Encryptor,
#                   relin_keys: seal.RelinKeys,
#                   scale: float):
#     """Evaluate a polynomial function of a ctxt, f(ctxt), for the given coefficients.

#     parameters
#     ----------
#     ctx: operand
#     coeffs: coefficients
#     evaluator: Evaluator module of the current SEAL context
#     encoder: Encoder module of the current SEAL context
#     relin_keys: Keys used to relinearize multiplied ctxts
#     scale: scale of the operand ctxt
#     """
    
#     degree = len(coeffs) - 1
#     plain_coeffs = coeffs_to_plaintext(coeffs, encoder, scale, broadcast=True)
#     powers = compute_all_powers(ctx, degree, evaluator, relin_keys)
#     output = multiply_and_add_coeffs(powers, plain_coeffs, coeffs, evaluator, scale)
    
#     return output


class CommonAlgorithms():
    """Wrapper for basic FHE algorithms.
        Supports both HEAAN and SEAL libraries.
    """
    def __init__(self, scheme, scheme_name):
        self.scheme_name = scheme_name
        self.scheme = scheme
        if scheme_name == "HEAAN":
            try:
                self.algo = he.SchemeAlgo(scheme._scheme)
            except:
                self.algo = he.SchemeAlgo(scheme)


    def function_poly(self, coeffs, ctx):
        """wrapper of polynomial evaluation functions of HEAAN and SEAL
        """
        if self.scheme_name == "HEAAN":
            output = he.Ciphertext()
            self.algo.function_poly(output, 
                       ctx, 
                       he.Double(coeffs), 
                       self.scheme.parms.logp, 
                       len(coeffs)-1)
            return output
        elif self.scheme_name == "SEAL":
            evaluator = self.scheme._evaluator
            encoder = self.scheme._encoder
            relin_keys = self.scheme.relin_keys
            scale = self.scheme.scale
            return seal_polyeval(ctx, coeffs, evaluator, encoder, relin_keys, scale)

    def exponential(self, ctx):
        """Evaluate exponential as polynomial approximation
        """
        coeffs = [1,1,0.5,1./6,1./24,1./120,1./720,1./5040,1./40320,1./362880,1./3628800]
        return self.function_poly(coeffs, ctx)

    def sigmoid(self, ctx):
        """Evaluate sigmoid as polynomial approximation
        """
        coeffs = [1./2,1./4,0,-1./48,0,1./480,0,-17./80640,0,31./1451520,0]
        return self.function_poly(coeffs, ctx)

    def logarithm(self, ctx):
        """Evaluate logarithm as polynomial approximation
        """
        coeffs = [0,1,-0.5,1./3,-1./4,1./5,-1./6,1./7,-1./8,1./9,-1./10]
        return self.function_poly(coeffs, ctx)

    def tanh(self, ctx):
        """Evaluate hyperbolic tangent as polynomial approximation
        """
        coeffs = [0,  4.758149e+00,  0, -1.838215e+01, 0,  3.860479e+01,  0, -3.728043e+01, 0,  1.331208e+01]
        return self.function_poly(coeffs, ctx)

    def ReLU(self, ctx):
        """Evaluate ReLU as 2-nd order polynomial approximation
        """
        coeffs = [0.47, 0.5, 0.09]
        return self.function_poly(coeffs, ctx)

    def _mat_mult_he(self, diagonals, ctx):
        """perform (Halevi-Shoup) matrix - vector multiplication using HEAAN

        parameters
        ----------
        diagonals: diagonalized (differentially rotated) ctxts

        NOTE
        ----
        diagonal are expected to be in appropriate mods
        """
        scheme = self.scheme
        n = self.scheme.parms.n
        logp = self.scheme.parms.logp
        #logq = self.parms.logq

        ctx_copy = he.Ciphertext()
        ctx_copy.copy(ctx)
        
        for i, diagonal in enumerate(diagonals):
            if i > 0: scheme.leftRotateFastAndEqual(ctx_copy, 1) # r = 1

            # Multiply with diagonal
            dd = he.Ciphertext()
            scheme.mult(dd, diagonal, ctx_copy)
            scheme.reScaleByAndEqual(dd, logp)
            # Reduce the scale of diagonal
            if i == 0:
                mvec = np.zeros(n)
                temp = he.Ciphertext()
                scheme.encrypt(temp, he.Double(mvec), n, logp, ctx_copy.logq - logp)
            
            # match scale 
            scheme.addAndEqual(temp, dd)

            del dd
        del ctx_copy
        return temp


    def _mat_mult_seal(self, diagonals, ctx):
        """perform (Halevi-Shoup) matrix - vector multiplication using SEAL

        parameters
        ----------
        diagonals: diagonalized (differentially rotated) ctxts

        NOTE
        ----
        diagonal are expected to be in appropriate mods
        """
        evaluator = self.scheme._evaluator 
        galois_keys = self.scheme.galois_keys
        output = seal.Ciphertext()
        
        for i in range(len(diagonals)):    
            temp = seal.Ciphertext()
            diagonal = diagonals[i]
            
            evaluator.rotate_vector(ctx, i, galois_keys, temp)
                
            evaluator.mod_switch_to_inplace(diagonal, temp.parms_id())
            evaluator.multiply_plain_inplace(temp, diagonal)
            evaluator.rescale_to_next_inplace(temp)
            
            if i == 0:
                output = temp
            else:
                evaluator.add_inplace(output, temp)

        return output

    def mat_mult_diag(self, diagonals, ctx):
        """ (Halevi-Shoup) matrix-vector multiplication

        parameters
        ----------
        diagonals: diagonalized (differentially rotated) ctxts
        """
        if self.scheme_name == "HEAAN":
            return self._mat_mult_he(diagonals, ctx)
        elif self.scheme_name == "SEAL":
            return self._mat_mult_seal(diagonals, ctx)

    def _sum_reduce_he(self, ctx, nslots, scheme):
        """return the sum of a Ciphertext (repeated nslot times), HEAAN version
        
        parameters
        ----------
        ctx: ciphertext
        nslots: number of slots of the ciphertext 
        scheme: scheme of the current HEAAN context

        example
        -------
        sum_reduct([1,2,3,4,5])
        >> [15,15,15,15,15]
        """
        output = he.Ciphertext()
        
        for i in range(nslots):
            
            if i == 0:
                temp = he.Ciphertext(ctx.logp, ctx.logq, ctx.n)
                scheme.leftRotateFast(temp, ctx, 2**i)
                scheme.add(output, ctx, temp)
            else:
                scheme.leftRotateFast(temp, output, 2**i)
                scheme.addAndEqual(output, temp)
        return output

    # def _sum_reduce_seal(self, ctx: seal.Ciphertext, n=None):
    #     """return the sum of a Ciphertext (repeated nslot times), SEAL version
    #     """        
    #     evaluator = self.scheme._evaluator 
    #     galois_keys = self.scheme.galois_keys
    #     if n is None:
    #         n = int(np.log2(self.scheme.nslots))

    #     temp = seal.Ciphertext()
    #     output = seal.Ciphertext()

    #     for i in range(n):
    #         if i == 0:
    #             temp = evaluator.rotate_vector(ctx, 2**i, galois_keys)
    #             output = evaluator.add(ctx, temp)
    #         else:
    #             temp = evaluator.rotate_vector(output, 2**i, galois_keys)
    #             evaluator.add_inplace(output, temp)
    #     return output


    def reduce(self, outputs, n=None):
        """warpper for sum-reduce function for HEAAN and SEAL
        """
        if self.scheme_name == "HEAAN":
            logp = self.scheme.parms.logp
            scheme = self.scheme

            for i, output in enumerate(outputs):
                output = self._sum_reduce_he(output, self.scheme.parms.logn, self.scheme._scheme)
                mask = np.zeros(self.scheme.parms.n)
                mask[0] = 1
                mask_hedb = he.ComplexDouble(mask)
                if i == 0:
                    res = he.Ciphertext()
                    scheme.multByConstVec(res, output, mask_hedb, logp)
                    scheme.reScaleByAndEqual(res, logp)
                else:
                    temp = he.Ciphertext()
                    scheme.multByConstVec(temp, output, mask_hedb, logp)
                    scheme.reScaleByAndEqual(temp, logp)
                    scheme.rightRotateFastAndEqual(temp, i)
                    scheme.addAndEqual(res, temp)

            return res
    
        elif self.scheme_name == "SEAL":
            return self._sum_reduce_seal(outputs, n=n)

    def dot_product_plain(self, ctx, ptx):
        """Computes the dot product between a ciphertext and a plaintext
            ptx be either SEAL plain text of HEAAN Double
        """

        output = self.scheme.multByConst(ctx, ptx)
        return self.reduce(output)
        
    def average(self, ctx, nctx):
        """return the mean of a ciphertext"""
        return self.reduce(ctx)/nctx
