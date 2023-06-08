from fase import heaan_loader
he = heaan_loader.load()
import numpy as np

class Scheme():
    """Common interface of HEAAN(ckks), SEAL(ckks, bfv) and PALISADE(bgv)

    Attributes
    ----------
    encrypt

    Notes
    -----
    This will need to be split into client-side and server-side, where server-side doesn't hold the secret key.
    """
    def __init__(self, name, parms, is_client=True):
        """
        
        """
        self.is_client = is_client
        self.sk = None
        if name in ['he', 'heaan', 'HEAAN']:
            logI = 4
            logT = 2
            ring = he.Ring()
            self.ring = ring
            if self.is_client:
                sk = he.SecretKey(ring)
                scheme = he.Scheme(sk, ring)
                scheme.addLeftRotKeys(sk)
                scheme.addRightRotKeys(sk)
                scheme.addBootKey(sk, parms.logn, parms.logq + parms.logI)
                #scheme.addBootKey(sk, parms['logn'], parms['logq'] + logI)
                self.sk = sk
            else:
                raise NotImplementedError
                # scheme WITHOUT the secret key
            self.algo = he.SchemeAlgo(scheme)
            self.__name = 'heaan' 
            
        elif name in ['seal', 'SEAL', 'se']:
            self.__name = 'seal' 
            raise NotImplementedError
        elif name in ['bfv', 'BFV']:
            self.__name = 'bfv' 
            raise NotImplementedError
        elif name in ['bgv', 'BGV']:
            self.__name = 'bgv' 
            raise NotImplementedError
        
        self.scheme = scheme
        self.parms = parms

    def encrypt(self, val, parms=None):
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
        if parms == None:
            parms = self.parms
        n = parms.n
        logp = parms.logp
        logq = parms.logq

        if self.__name == 'heaan':
            assert len(val) <= n, f"the array is longer than #slots: len(val) = {len(val)}, n = {n}"
            ctxt = he.Ciphertext()#logp, logq, n)
            vv = np.zeros(n) # Need to initialize to zero or will cause "unbound"
            vv[:len(val)] = val
            self.scheme.encrypt(ctxt, he.Double(vv), n, logp, logq)
            del vv
        return ctxt

    def encode(self, val, parms=None):
        """Encode an array/list of numbers to a Plaintext
        pass 'n' if val is a vector.
        """
        if parms == None:
            parms = self.parms
        n = parms.n
        logp = parms.logp
        logq = parms.logq

        ptxt = he.Plaintext(logp, logq, n)# beware signature order. 
        vv = np.zeros(n) # Need to initialize to zero or will cause "unbound"
        vv[:len(val)] = val
        self.scheme.encode(ptxt, he.Double(val), n, logp, logq)
        del vv 
        return ptxt

    def print_dec(self, x):
        """print decrypted value
        """
        try:
            print(self.scheme.decrypt(self.sk, x))
        except AttributeError as err:
            print("Do you have a secret key?")
            raise
            
    def decrypt(self, ctxt):
        """Decrypt a ciphertext.
        """
        if self.sk is not None:
            dd = self.scheme.decrypt(self.sk, ctxt)
            arr = np.zeros(ctxt.n, dtype=np.complex128)
            dd.__getarr__(arr)
            del dd
            return arr
        else:
            print("you can't decrypt without a secret key")

    def bootstrap_inplace(self, ctxt):
        """Bootstrap in place
        """
        self.scheme.bootstrapAndEqual(ctxt, 
                                    self.parms.logq,
                                    self.parms.logQ,
                                    self.parms.logT,
                                    self.parms.logI)
    
    def bootstrap(self, ctxt):
        c_new = he.Ciphertext()
        self.scheme.bootstrapAndEqual(c_new,
                                    ctxt, 
                                    self.parms.logq,
                                    self.parms.logQ,
                                    self.parms.logT,
                                    self.parms.logI)
        return c_new


class HEAANParameters_specific():
    """Setup HEAAN FHE parameters

        Parameters
        ----------
        e.g.) 12345.678 -> logq = 8, logp = 3 =?= pBits

        logn (bBits) : log(Number of slots per ciphertext). 
               If logn = 10, one ciphertext can hold 1024 values 
               and perform 1024-way SIMD computation
        logp : 
        
        wBits :
        
        pBits : 
        
        lBits :
        
        logT : Bootstrapping parameter.  
        
        logI : Bootstrapping parameter. 

        iterPerBoot: A temporary, application specific parameter. 
                     It depdens on the depth of multiplication per iteration. 
                     Note that for a deep NN, more than one bootstrapping may be required per iteration. 
        
        """

    def __init__(self, logn, logp, logq,
                 wBits=40, 
                 pBits=15, 
                 lBits=5, 
                 logT=3, 
                 logI=4,
                 iterPerBoot=2):
        self.scheme_name = "HEAAN"
        self.logn = logn
        self.n = 2**logn
        self.logp = logp
        self.logq = logq
        self.logT = logT
        self.logI = logI
        self.wBits = wBits 
        self.pBits = pBits
        self.lBits = lBits
        self.iterPerBoot=iterPerBoot
        #self._cal_boot_params()
        self.cparams = {'n':self.n,
                        'logp':self.logp,
                        'logq':logq}
                        #'logq':self.logq}
    
    def _cal_boot_params(self):
        logI = self.logI
        logT = self.logT
        wBits = self.wBits
        lBits = self.lBits
        pBits = self.pBits
        
        # model mod reduction in an iteration of the application.
        self.logQ = wBits + lBits + self.iterPerBoot * (3 * wBits + 2 * pBits)

        self.logq = wBits + lBits
        bitForBoot = 16 + logT + logI + (logI + logT + 6) * (self.logq + logI)
        self.logQBoot = self.logQ + bitForBoot

        NBnd = np.ceil(self.logQBoot * (80 + 110) / 3.6)
        logNBnd = np.log2(NBnd)
        logN = int(np.ceil(logNBnd))
        self.logN = logN