# from cv2 import split
import numpy as np
import os 
#from fase.core.heaan import he
from fase.heaan_loader import load
he = load()

class HEAANContext():
    def __init__(self, 
                scale_bit: int,
                logp: int,
                logq: int,
                is_owner=True,
                rot_l = None,
                rot_r = None,
                boot=True,
                key_path="./",
                load_sk=False,
                logqBoot=None,
                FN_SK="SecretKey.txt",
                FN_ENC="EncKey.txt",
                FN_MUL="MulKey.txt",
                FN_ROT="RotKey.txt",
                FN_CONJ="ConjKey.txt"):
        """Setup a HEAAN context and initialize all the relevant attributes.

            Parameters
            ----------
            scale_bit: int
                Number of scale bits. 'logn'
                2**logn = number of ciphertext slots
            logp: int
            logq: int

            is_owner: bool
                Client has the secret key, while sever only can access to the public key 
                provided by the client.

            rot_l: left rotation shifts.  None, (list of) integers, or "all"
            rot_r: right rotation shifts. None, (list of) integers, or "all"

            boot: bool
                Generate bootstrapping key (, which requires all the rotation keys)
            key_path: string
                path to keys
            load_sk: bool
                load secretkey. Effective only when is_owner==True 
            

            Attributes
            ----------
            ring: HEAAN Ring

            scheme: HEAAN Scheme 
                Performs essential operations such as encode, encrypt, decrypt, bootstrap, add, multiply, rotate,..
            
            algo: HEAAN SchemeAlgo
        """

        self._name = "HEAAN" 
        self.parms = HEAANParameters(scale_bit, logp, logq, logT=2, logI=4)
        self._is_owner = is_owner
        self._key_path = key_path
        self._boot = boot
        self.logqBoot = logqBoot
        
        self._lkey=self.parse_rotkey(rot_l)
        self._rkey=self.parse_rotkey(rot_r)

        self.ring = he.Ring()
        if self._is_owner:
            print("Initializing the scheme as the data owner", flush=True)
            if (not os.path.exists(self._key_path+FN_SK) or \
                not os.path.exists(self._key_path+FN_ENC) or \
                 not os.path.exists(self._key_path+FN_MUL)):
                load_sk = False
            if load_sk:
                print("Loading a secret key from: ", self._key_path+FN_SK, flush=True)
                self.sk = he.SecretKey(self._key_path+FN_SK)
                
                try:# Load if matching keys are stored
                    self._scheme = he.Scheme(self.ring, True, self._key_path)
                    if rot_l:
                        self.loadLKey(self._lkey)
                    if rot_r:
                        self.loadRKey(self._rkey)
                    if boot:
                        self._scheme.loadBootKey(self.parms.logn, self.parms.logq + self.parms.logI)
                        print("loading boot keys done", flush=True)
                except:
                    # Or generated other keys with the loaded SK
                    self._scheme = he.Scheme(self.sk, self.ring, True, self._key_path)
                    print("Failed to load matching public keys... generating them...", flush=True)
                    if rot_l is not None: 
                        self.addLkey(self._lkey)
                    if rot_r is not None: 
                        self.addRkey(self._rkey)
                    if boot: 
                        self._scheme.addBootKey(self.sk, self.parms.logn, self.parms.logq + self.parms.logI)                        
            else:
                self.sk = he.SecretKey(self.ring, self._key_path+FN_SK)
                self._scheme = he.Scheme(self.sk, self.ring, True, self._key_path)
                if rot_l is not None: 
                    self.addLkey(self._lkey)
                if rot_r is not None: 
                    self.addRkey(self._rkey)
                if boot: 
                    self._scheme.addBootKey(self.sk, self.parms.logn, self.parms.logq + self.parms.logI)
            
        else:
            self._scheme = he.Scheme(self.ring, True, self._key_path)
            if rot_l:
                self.loadLKey(self._lkey)
            if rot_r:
                self.loadRKey(self._rkey)
            print("KEY LOADING COMPLETE")

        self.algo = he.SchemeAlgo(self._scheme)
        print("HEAAN CKKS setup is ready ")

    @staticmethod
    def gen_2_exp(n):
        return [2**nn for nn in range(n+1)]

    def parse_rotkey(self, rotkey):
        """Figure out which rotation keys to use

        parameters
        ----------
        rotkey: 
        """
        if rotkey is None:
            return []
        try:
            iter(rotkey)
            try: 
                if rotkey.lower() == "all":
                    return HEAANContext.gen_2_exp(self.parms.logn)
            except: 
                if all(isinstance(x, int) for x in rotkey):
                    return rotkey
        except:
            if isinstance(rotkey, int): return [rotkey]
            print("rotkey provided:", rotkey)
            raise ValueError("Rotation key not understood -- None," 
                            "'all', or an iterable of integers are expected")
                

    def addLkey(self, r=None):
        """ Generate Left rotation keys for r

        parameters
        ----------
        r: list of rotation shifts in integer

        NOTE
        ----
        Only use individual key add, not addLetfRotKey's'()
        """
        #if r is None:
        #    self.scheme.addLeftRotKeys(self.sk)
        #else:
        for rr in r:
            self._scheme.addLeftRotKey(self.sk, rr)

    def addRkey(self, r=None):
        """ Generate Right rotation keys for r

        parameters
        ----------
        r: list of rotation shifts in integer

        NOTE
        ----
        Only use individual key add, not addRightRotKey's'()
        """
        for rr in r:
            self._scheme.addRightRotKey(self.sk, rr)

    def loadLKey(self, r=None):
        """Load left rotation key
        
        todo 
        -----
        Assert if Rot key is already available.
        """
        self._has_lkeys=[]
        for rr in r:
            self._scheme.loadLeftRotKey(rr)
            self._has_lkeys.append(rr)

    def loadRKey(self, r=None):
        """Load right rotation key
        
        todo 
        -----
        Assert if Rot key is already available.
        """
        self._has_rkeys=[]
        for rr in r:
            self._scheme.loadRightRotKey(rr)
            self._has_rkeys.append(rr)


    def encrypt(self, val, n=None, logp = None, logq = None):
        """Encrypt an array/list of numbers to a cipher text

        parameters
        ----------
        val: ndarray / list
        parms: (optional) CKKS parameter instance
        
        Notes
        -----
        HEAAN Double array needs to initialize to zeros, or garbage values may cause 
        "RR: conversion of a non-finite double" error.
        """
        if n is None:
            n = self.parms.n
        if logp is None:
            logp = self.parms.logp
        if logq is None:
            logq = self.parms.logq
    
        assert len(val) <= n, f"the array is longer than #slots: len(val) = {len(val)}, n = {n}"
        ctxt = he.Ciphertext()#logp, logq, n)
        vv = np.zeros(n) # Need to initialize to zero or will cause "unbound"
        vv[:len(val)] = val
        self._scheme.encrypt(ctxt, he.Double(vv), n, logp, logq)
        del vv
        return ctxt

    def encode(self, val, parms=None):
        """Encode an array/list of numbers to a Plaintext

        parameters
        ----------
        val: sequence of numbers
            First converted to ndarray and then passed to FHE encode function.
            requires that len(val) <= # slots 
        """
        if parms == None:
            parms = self.parms
        n = parms.n
        logp = parms.logp
        logq = parms.logq

        ptxt = he.Plaintext(logp, logq, n)# beware signature order. 
        vv = np.zeros(n) # Need to initialize to zero or will cause "unbound"
        vv[:len(val)] = val
        self._scheme.encode(ptxt, he.Double(val), n, logp, logq)
        del vv 
        return ptxt

    def print_dec(self, ctxt, n):
        """print decrypted value

        parameters
        ----------
        ctxt: HEAAN Ciphertext
        n: int
            Number of slots to print
        """
        dd = self.decrypt(ctxt)
        print(dd[:n])
            
    def decrypt(self, ctxt, complex=False):
        """Decrypt a ciphertext
        
        parameters
        ----------
        ctxt: HEAAN Ciphertext
        complex:bool
            Whether to return as double or complex number
        """
        if self.sk is not None:
            dd = self._scheme.decrypt(self.sk, ctxt)
            arr = np.zeros(ctxt.n, dtype=np.complex128)
            dd.__getarr__(arr)
            del dd
            if complex: return arr
            else: return arr.real
        else:
            print("you can't decrypt without a secret key")

    def bootstrap(self, ctxt, logq, inplace=True):
        """Bootstrap 

        parameters
        ----------
        ctxt: HEAAN Ciphertext
        inplace: bool [True]
        """
        if inplace:
            self._scheme.bootstrapAndEqual(ctxt, 
                                    logq,
                                    self.parms.logQ,
                                    self.parms.logT,
                                    self.parms.logI)
            print("bootstrap done")            

    def add(self, ctxt1, ctxt2, inplace=False):
        """Add ctxt2 to ctxt1

        parameters
        ----------
        ctxt1: HEAAN Ciphertext
        ctxt2: HEAAN Ciphertext
        inplace: bool [False]
        """
        if inplace:
            self._scheme.addAndEqual(ctxt1, ctxt2)
        else:
            new_ct = he.Ciphertext()
            self._scheme.add(new_ct, ctxt1, ctxt2)
            return new_ct

    def addConst(self, ctxt, val, inplace=False):
        """

        Todo
        ----
        1. Why take a sequence when only scalar input makes sense. 
        change something like
        > self._scheme.addConstAndEqual(ctxt, he.Double([val]), ctxt.logp)

        2. Need to write AddConstVec, AddConstVecAndEqual functions.

        """
        if inplace:
            self._scheme.addConstAndEqual(ctxt, he.Double(list(val)), ctxt.logp)
        else:
            new_ct = he.Ciphertext()
            self._scheme.addConst(new_ct, ctxt, he.Double(list(val)), ctxt.logp)
            return new_ct

    def sub(self, ctxt1, ctxt2, inplace=False):
        """Sbutract ctxt2 from ctxt1

        parameters
        ----------
        ctxt1: HEAAN Ciphertext
        ctxt2: HEAAN Ciphertext
        inplace: bool [False]
        """
        if inplace:
            self._scheme.subAndEqual2(ctxt1, ctxt2)
        else:
            new_ct = he.Ciphertext()
            self._scheme.sub(new_ct, ctxt1, ctxt2)
            return new_ct      

    def multByConst(self, ctxt, val, inplace=False, rescale=False):
        """Multiply (all slots of) a ciphertext by a plain value.
        
        parameters
        ----------
        ctxt: HEAAN Ciphertext
        val: sequence of numbers
            First converted to ndarray and then passed to FHE encode function.
            requires that len(val) <= # slots
        inplace: bool [False] 
        """
        if inplace:
            self._scheme.multByConstAndEqual(ctxt, he.Double(list(val)), ctxt.logp)
            if rescale: self.rescale(ctxt)
        else:
            new_ct = he.Ciphertext()
            self._scheme.multByConst(new_ct, ctxt, he.Double(list(val)), ctxt.logp)
            if rescale: self.rescale(new_ct)
            return new_ct

    def multByVec(self, ctxt, val, inplace=False, rescale=False):
        """Element-wise multiplication of a ciphertext and a plain sequence
        
        parameters
        ----------
        ctxt: HEAAN Ciphertext
        val: sequence of numbers
            First converted to ndarray and then passed to FHE encode function.
            requires that len(val) <= # slots
        inplace: bool [False]
        """
        try:
            iter(val)
        except ValueError as err:
            print("Need an iterable", err)
            
        if inplace:
            self._scheme.multByConstVecAndEqual(ctxt, he.Double(val), ctxt.logp)
            if rescale: self.rescale(ctxt)
        else:
            new_ct = he.Ciphertext()
            self._scheme.multByConstVec(new_ct, ctxt, he.Double(val), ctxt.logp)
            if rescale: self.rescale(new_ct)
            return new_ct

    def mult(self, ctxt1, ctxt2, inplace=False, rescale=False):
        """Element-wise multiplication of two ciphertexts
        
        parameters
        ----------
        ctxt1: HEAAN Ciphertext
        ctxt2: HEAAN Ciphertext
        inplace: bool [False]
        """
        if inplace:
            self._scheme.multAndEqual(ctxt1, ctxt2)
            if rescale: self.rescale(ctxt1)
        else:
            new_ct = he.Ciphertext()
            self._scheme.mult(new_ct, ctxt1, ctxt2)
            if rescale: self.rescale(new_ct)
            return new_ct

    def square(self, ctxt, inplace=False):
        """Element-wise square of a ciphertext
            
        parameters
        ----------
        ctxt: HEAAN Ciphertext
        inplace: bool [False]
        """
        if inplace:
            self._scheme.squareAndEqual(ctxt)
        else:
            new_ct = he.Ciphertext()
            self._scheme.square(new_ct, ctxt)
            return new_ct

    def rescale(self, ctxt, scale=None):
        """Rescale ciphertext
        
        parameters
        ----------
        ctxt: HEAAN Ciphertext
        scale: intenger [None]

        If scale == None, reduce the scale by HEAANContext.logp
        """
        if scale is None:
            self._scheme.reScaleByAndEqual(ctxt, ctxt.logp - self.parms.logp)
        else:
            self._scheme.reScaleToAndEqual(ctxt, scale)

    def match_mod(self, ctxt, target):
        """Switch mod of ctxt down to target.logq

        parameters
        ----------
        ctxt: HEAAN Ciphertext 
        target: HEAAN Ciphertext
        """
        self._scheme.modDownToAndEqual(ctxt, target.logq)

    def lrot(self, ctxt, r, inplace=False):
        """Left-rotate by r
        or, bring element at index r to index 0.
        """
        if r < 0:
            r = self.parms.n - r
        
        if inplace:  
            for rr in HEAANContext.split_in_twos(r):
                self._scheme.leftRotateFastAndEqual(ctxt, rr)
        else:
            new_ctxt = None
            for rr in HEAANContext.split_in_twos(r):
                if new_ctxt is None:
                    new_ctxt = he.Ciphertext()
                    self._scheme.leftRotateFast(new_ctxt, ctxt, rr)
                else:
                    self._scheme.leftRotateFastAndEqual(new_ctxt, rr)
            return new_ctxt

    def rrot(self, ctxt, r, inplace=False):
        """Right-rotate by r
        """
        r = self.parms.n - r
        for rr in HEAANContext.split_in_twos(r)[:-1]:
            print(rr)
            self._scheme.leftRotateFastAndEqual(ctxt, rr)
        
        rr = HEAANContext.split_in_twos(r)[-1]
        if inplace:    
            self._scheme.leftRotateFastAndEqual(ctxt, rr)
        else:
            new_ctxt = he.Ciphertext()
            self._scheme.leftRotateFast(new_ctxt, ctxt, rr)
            return new_ctxt

    @staticmethod
    def split_in_twos(val):
        """Split a number into a list of 2**n"""
        nums = []
        for d in bin(val)[2:]:
            nums = [nn << 1 for nn in nums]
            if int(d) == 1: nums.append(int(d))

        return nums



class HEAANParameters():
    """Setup HEAAN FHE parameters

        Parameters
        ----------
        logn (bBits) : log2(N slots). 
            e.g., If logn = 10, one ciphertext can hold 1024 values 
            and perform 1024-way SIMD computation
        
        logp : scale (precision) of Ciphertext 
        logT : Bootstrapping parameter.  
        logI : Bootstrapping parameter. 
        mult_depth: Number of depth allowed per bootstrap

        NOTE
        ----
        Requires logq > logp * mult_depth and logq' > logq * mult_depth, 
        where logq' is the logq after bootstrapping.
        
        """

    def __init__(self, logn, logp, logq,
                 logT=3, 
                 logI=4,
                 logQ=800,
                 mult_depth=12):
        self.scheme_name = "HEAAN"
        self.logn = logn
        self.n = 2**logn
        self.logp = logp
        self.logq = logq
        self.logT = logT
        self.logI = logI
        self.mult_depth=mult_depth
        self.logQ = logQ
        self.cparams = {'n':self.n,
                        'logp':self.logp,
                        'logq':logq}

