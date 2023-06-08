from fase import seal
import numpy as np
from typing import DefaultDict, List
from abc import ABC, abstractclassmethod

class SEALBFVContext():
    """Setup a SEAL context and initialize all the relevant attributes.

        Most of the 'setup' stage is routine. 
        But poly_modulus_degree and coeff_moduli must be correctly designed for the 
        problem at hand. 
        Note that by default it assumes tc128 security level.
        Note that SEAL BFV and CKKS have many methods in common.


        Parameters
        ----------
        poly_modulus_degree : int
            Polynomial modulus degree. eg., 8192

        coeff_moduli : list of int
            coefficient modulus as a list of ints.
            * the number of coeff_moduli = depth of allowed multiplications
        
        scale_bit : int

        plain_modulus : int or asttribute of seal.PlainModulus
            Batching requires 'seal.PlainModulus.Batching()' 

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
    def __init__(self, poly_modulus_degree: int,
                       prime_bit: int = 20,
                       coeff_moduli: List[int] = [],
                       is_client=True
                       ):
        
        self._name = "SEAL"
        self.parms = seal.EncryptionParameters(seal.scheme_type.bfv)
        self.parms.set_poly_modulus_degree(poly_modulus_degree)
        
        if len(coeff_moduli) == 0: 
            print("[SEALBFVContext] coeff_moduli is not given. Defaulting to a suggested set")
            self.parms.set_coeff_modulus(seal.CoeffModulus.BFVDefault(poly_modulus_degree))
        else:
            self.parms.set_coeff_modulus(seal.CoeffModulus.Create(
                                        poly_modulus_degree, coeff_moduli))
        #self.scale = int(self.parms.poly_modulus_degree()/2)
        self.parms.set_plain_modulus(seal.PlainModulus.Batching(poly_modulus_degree, prime_bit))
        self.context = seal.SEALContext(self.parms)#, True, seal.sec_level_type.tc128)
        self.keygen = seal.KeyGenerator(self.context)

        self.public_key = self.keygen.create_public_key()
        self.relin_keys = self.keygen.create_relin_keys()
        self.galois_keys = self.keygen.create_galois_keys()

        self._encoder = seal.BatchEncoder(self.context)
        self.nslots = self._encoder.slot_count()
        if is_client:
            self.sk = self.keygen.secret_key()
            self._encryptor = seal.Encryptor(self.context, self.public_key)
        else:
            self._encryptor = seal.Encryptor(self.context, self.public_key)
        if is_client:
            self._decryptor = seal.Decryptor(self.context, self.sk)
        self._evaluator = seal.Evaluator(self.context)
        self.algo = None # Advanced operators to be implemented
        
        print("SEAL BFV scheme is ready")

    def encode(self, val):
        return self._encoder.encode(val)

    def decode(self, val):
        return self._encoder.decode(val)

    def encrypt(self, val):
        ptx = self.encode(val)
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

    def add_plain(self, ctxt, ptxt, inplace=False):
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

    def sub_plain(self, ctxt, ptxt, inplace=False):
        if inplace:
            self._evaluator.sub_plain_inplace(ctxt, ptxt)
        else:
            return self._evaluator.sub_plain(ctxt, ptxt)

    def multByConst(self, ctxt, val, inplace=False, broadcast=False):
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
            val = list(val)

        if broadcast: 
            assert len(val) == 1
            val = np.repeat(val, self.nstlos)

        ptxt = self.encode(val)
        if inplace:
            self._evaluator.multiply_plain_inplace(ctxt, ptxt)
        else:
            return self._evaluator.multiply_plain(ctxt, ptxt)

    def mult(self, ctxt1, ctxt2, inplace=False):
        """Multiply two ciphertexts
        """
        if inplace:
            self._evaluator.multiply_inplace(ctxt1, ctxt2)
        else:
            return self._evaluator.multiply(ctxt1, ctxt2)

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
        """Rescale ciphertext to next coeff moduli 
        and manually fix small deviation by default.
        """
        self._evaluator.rescale_to_next_inplace(ctxt)
        if scale is None:
            scale = self.scale
        ctxt.scale(scale)

    def match_mod(self, ctxt, target):
        """Switch ctxt's mod down to target's 
        """
        self._evaluator.mod_switch_to_inplace(ctxt, target.parms_id())

    def rot(self, ctxt, r: int, inplace=False):
        """rotate left by r (negative r for right rotation)
        """
        if inplace:
            self._evaluator.rotate_rows_inplace(ctxt, r, self.galois_keys)    
        else:
            return self._evaluator.rotate_rows(ctxt, r, self.galois_keys)

    def exponentiate(self, ctxt, r, inplace=False):
        if inplace:
            self._evaluator.exponentiate_inplace(ctxt, r, self.relin_keys)
        else:
            return self._evaluator.exponentiate(ctxt, r, self.relin_keys)