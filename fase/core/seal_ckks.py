from fase import seal
import numpy as np
from typing import List
from abc import ABC, abstractclassmethod

class SEALContext():
    """Setup a SEAL context and initialize all the relevant attributes.

        Most of the 'setup' stage is routine. 
        But poly_modulus_degree and coeff_moduli must be correctly designed for the 
        problem at hand. 
        Note that by default it assumes tc128 security level.


        Parameters
        ----------
        poly_modulus_degree : int
            Polynomial modulus degree. eg., 8192

        coeff_moduli : list of int
            Number of coeff_moduli = depth of allowed multiplications
        scale_bit : int

        is_client : bool
            Client has the secret key, while sever only can access to the public key 
            provided by the client.


        Attributes
        ----------
        parms : Seal EncryptionParameters

        context : Seal SEALContext

        encoder : Seal CKKSEncoder
            Encode / Decode plain value to/from plaintext

        encryptor : Seal Encrypter
            encrypts plaintext to ciphertext

        decryptor : Seal Decryptor
            Decryps ciphertexts to plaintexts. Only available for client. 

        evaluator : Seal Evaluator
            This module performs all the coputations among plaintexts and ciphertexts --
            Add, Multiply, Rotate, Modulus switch, and Rescaling operations.
    """
    def __init__(self, poly_modulus_degree: int = None,
                       coeff_moduli: List[int] = None,
                       scale_bit: int = None,
                       is_client=True
                       ):
        
        self._name = "SEAL"
        self._fn_secret = "seal_sk.dat"
        self._fn_relin = 'seal_relin.dat'
        self._fn_galois = 'seal_rot.dat'
        self._fn_pub = "seal_pub.dat"
        self._fn_parms = "seal_parms.dat"


        self.parms = seal.EncryptionParameters(seal.scheme_type.ckks)
        if is_client:
            self.parms.set_poly_modulus_degree(poly_modulus_degree)
        
            self.parms.set_coeff_modulus(seal.CoeffModulus.Create(
                poly_modulus_degree, coeff_moduli))
            self.parms.save(self._fn_parms)
        else:
            self.parms.load(self._fn_parms)

        
        self.nslots = int(self.parms.poly_modulus_degree()/2)
        self.scale = np.power(2,scale_bit)
        
        self.context = seal.SEALContext(self.parms, True, seal.sec_level_type.tc128)

        if is_client:
            self.keygen = seal.KeyGenerator(self.context)
            self.public_key = self.keygen.create_public_key()
            self.public_key.save(self._fn_pub)
            self.relin_keys = self.keygen.create_relin_keys()
            self.relin_keys.save(self._fn_relin)
            self.galois_keys = self.keygen.create_galois_keys()
            self.galois_keys.save(self._fn_galois)
        else:
            self.public_key = seal.PublicKey()
            self.public_key.load(self.context, self._fn_pub)
            self.relin_keys = seal.RelinKeys()
            self.relin_keys.load(self.context, self._fn_relin)
            self.galois_keys = seal.GaloisKeys()
            self.galois_keys.load(self.context, self._fn_galois)            

        self._encoder = seal.CKKSEncoder(self.context)
        if is_client:
            self.sk = self.keygen.secret_key()
            self.sk.save(self._fn_secret)
            self._encryptor = seal.Encryptor(self.context, self.public_key)
        else:
            self._encryptor = seal.Encryptor(self.context, self.public_key)
        if is_client:
            self._decryptor = seal.Decryptor(self.context, self.sk)
        self._evaluator = seal.Evaluator(self.context)
        self.algo = None # Advanced operators to be implemented
        
        print("SEAL CKKS scheme is ready")
        

    def encode(self, val, scale=None):
        """encode numbers into SEAL palin text
        """
        if scale is None:
            scale = self.scale
        return self._encoder.encode(val, scale)

    def encrypt(self, val, scale=None):
        """encrypt plain numbers into SEAL Ciphertext
        """
        ptx = self.encode(val, scale)
        return self._encryptor.encrypt(ptx) 

    def decrypt(self, ctxt):
        ptx = self._decryptor.decrypt(ctxt)
        return self._encoder.decode(ptx)

    def add(self, ctxt1, ctxt2, inplace=False):
        """Add two ciphertexts
        """
        if inplace:
            self._evaluator.add_inplace(ctxt1, ctxt2)
        else:
            return self._evaluator.add(ctxt1, ctxt2)

    def addConst(self, ctxt, val, inplace=False, broadcast=False):
        try:
            len(val)
        except:
            val = [val]

        if broadcast: 
            assert len(val) == 1
            val = np.repeat(val, self.nslots)
            
        ptxt = self._encoder.encode(val, ctxt.scale())
        self.match_mod(ptxt, ctxt)

        if inplace:
            self._evaluator.add_plain_inplace(ctxt, ptxt)
        else:
            return self._evaluator.add_plain(ctxt, ptxt)

    def sub(self, ctxt1, ctxt2, inplace=False):
        """Sbutract two ciphertexts
        """
        if inplace:
            self._evaluator.sub_inplace(ctxt1, ctxt2)
        else:
            return self._evaluator.sub(ctxt1, ctxt2)

    def multByConst(self, ctxt, val, inplace=False, broadcast=False, rescale=False):
        """Multiply a ciphertext by a plain value.
        
        Parameters
        ----------
            inplace : bool

            broadcast : bool
                multiply all slots of a ciphertext by a plain value. 
                plain value must be scalar (or 1-element list/array)
        """
        try:
            len(val)
        except:
            val = [val]

        if broadcast: 
            assert len(val) == 1
            val = np.repeat(val, self.nslots)

        ptxt = self.encode(val, ctxt.scale()) # 이거랑 안 맞출 이유가 있나..?
        self.match_mod(ptxt, ctxt)

        scale_org = ctxt.scale()
        if inplace:
            self._evaluator.multiply_plain_inplace(ctxt, ptxt)
            if rescale: self.rescale(ctxt, scale_org)
        else:
            tmp = self._evaluator.multiply_plain(ctxt, ptxt)
            if rescale: self.rescale(tmp, scale_org)
            return tmp

    def mult(self, ctxt1, ctxt2, inplace=False, relin=True):
        """Multiply two ciphertexts
        """
        assert ctxt1.scale() == ctxt2.scale(), "Ctxts scale mismatch"
        if inplace:
            self._evaluator.multiply_inplace(ctxt1, ctxt2)
            if relin: self.relin(ctxt1)
        else:
            ctxt_out = self._evaluator.multiply(ctxt1, ctxt2)
            if relin: self.relin(ctxt_out)
            return ctxt_out

    def square(self, ctxt, inplace=False):
        """Square of ciphertext
        """
        if inplace:
            self._evaluator.square_inplace(ctxt)
        else:
            return self._evaluator.square(ctxt)

    def relin(self, ctxt):
        """Relinearization is required after every multiplication
        """
        self._evaluator.relinearize_inplace(ctxt, self.relin_keys)

    def rescale(self, ctxt, scale=None):
        """Rescale ciphertext to next coeff modulus 
        and manually fix small deviation by default.
        """
        self._evaluator.rescale_to_next_inplace(ctxt)
        if scale is None:
            scale = self.scale
        ctxt.scale(scale)

    def match_mod(self, ctxt, target):
        """Switch txt's mod down to target's mod
        This method applies to both ctxt and ptxt. 
        Modulus mismatch bewteen a ctxt and a *ptxt* is not allowed, either.
        """
        self._evaluator.mod_switch_to_inplace(ctxt, target.parms_id())

    def lrot(self, ctxt, r, inplace=False):
        """Left-rotate by r
        or, bring element at index r to index 0.
        """
        if inplace:
            self._evaluator.rotate_vector_inplace(ctxt, r, self.galois_keys)    
        else:
            new_ctxt = self._evaluator.rotate_vector(ctxt, r, self.galois_keys)
            return new_ctxt

    def rrot(self, ctxt, r, inplace=False):
        """Right-rotate by r
        """
        return self.lrot(ctxt, -r, inplace=inplace)
