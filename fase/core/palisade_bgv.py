import numpy as np
from fase import pycrypto

class PalisadeBGVContext():
    """All inputs to Palisade BGV scheme are expected to be lists of integers. 
    This class will internally convert inputs to lists of integers if possible.
    
    NOTE
    ----
    Palisade's automatic rescaling mechanism is enabled by default (EXACTRESCALE), 
    and all the relevant methods are hidden in this class. 
    """
    def __init__(self, plaintext_modulus, multDepth, maxdepth,
                 sigma=3.2, rot=None, silent=False):
        self._rot_indices = []
        self._silent = silent
        self._scheme = pycrypto.BGVwrapper()
        self._scheme.KeyGen(multDepth,
                            plaintext_modulus,
                            sigma,
                            maxdepth, 
                            )
        
        # Generate rotation keys
        if rot is not None:
            self.add_rotKey(rot)
            
    def add_rotKey(self, rot):
        self._rot_indices = rot
        try:
            self._scheme.EvalAtIndexKeyGen(self._to_int_list(self._rot_indices))
        except ValueError:
            raise ValueError("Invalid rotation indices are given")
            
    def encrypt(self, val: list):
        return self._scheme.Encrypt(self._to_int_list(val))
    
    def decrypt(self, ctxt):
        return np.array(self._scheme.Decrypt(ctxt), dtype=np.int64)
    
    def add(self, ctxt1, ctxt2, inplace=False):
        if inplace:
            self._scheme.EvalAddInPlace(ctxt1, ctxt2)
        else:
            return self._scheme.EvalAdd(ctxt1, ctxt2)
    
    def sub(self, ctxt1, ctxt2, inplace=False):
        if inplace and not self._silent: print("'inplace' not supported!")
        return self._scheme.EvalSub(ctxt1, ctxt2)
    
    def mult(self, ctxt1, ctxt2, inplace=False):
        if inplace and not self._silent: print("'inplace' not supported!")
        return self._scheme.EvalMult(ctxt1, ctxt2)
    
    def multByVec(self, ctxt, ptxt: list, inplace=False):
        if inplace and not self._silent: print("'inplace' not supported!")
        return self._scheme.EvalMultConst(ctxt, self._to_int_list(ptxt))
    
    def rot(self, ctxt, r: int, inplace=False):
        """rotate left by r (negative r for right rotation)
        """
        assert r in self._rot_indices, f"rotation key for {r} not ready"
        if inplace and not self._silent: print("'inplace' not supported!")
        return self._scheme.EvalAtIndex(ctxt, r)
    
    
    @staticmethod
    def _to_int_list(val):
        """val must be a sequence of numbers that can 
        unambiguously be converted to a list of integers 
        """
        try:
            ll = [int(v) for v in val]
            if np.all(ll == val):
                return ll
            else:
                raise ValueError("Can't be converted to integer list")
        except ValueError:
            raise ValueError("Only a list of integers is allowed")
