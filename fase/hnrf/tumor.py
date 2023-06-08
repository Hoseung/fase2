import numpy as np
from time import time
from fase import HEAAN as he

#from fase.hnrf.tree import NeuralTreeMaker
from . import heaan_nrf 
from .hetree import HNRF

class Param():
    def __init__(self, n=None, logn=None, logp=None, logq=None, logQboot=None):
        self.n = n
        self.logn = logn
        self.logp = logp
        self.logq = logq 
        self.logQboot = logQboot
        if self.logn == None:
            self.logn = int(np.log2(n))

class HNRF_builder():
    def __init__(self, Nmodel, do_reduction=False, device="cpu"):
        
        self._do_reduction = do_reduction
        logq = 540
        logp = 30
        logn = 14
        n = 1*2**logn

        self.parms = Param(n=n, logp=logp, logq=logq)
        self.ring = he.Ring()
        self.secretKey = he.SecretKey(self.ring)
        self.scheme = he.Scheme(self.secretKey, self.ring, False)
        self.algo = he.SchemeAlgo(self.scheme)
        
        # reduction때는 right rotation N_class개 필요. 
        Nclass = len(Nmodel.head_bias)
        if self._do_reduction:
            self.scheme.addLeftRotKeys(self.secretKey)
            for i in range(Nclass):
                self.scheme.addRightRotKey(self.secretKey, i+1) # 
        else:
            # reduction 안 하면 하나짜리 rotation만 여러번 반복.
            self.scheme.addLeftRotKey(self.secretKey, 1)

        Nmodel.to_device(device)
        self.construct_hnrf(Nmodel, device=device)
            
            
    def construct_hnrf(self, Nmodel, device="cpu"):
        t0 = time()
        h_rf = HNRF(Nmodel, device)
        self.nrf_evaluator = heaan_nrf.HETreeEvaluator.from_model(h_rf,
                                                            self.scheme,
                                                            self.parms,
                                                            Nmodel.activation.coef,
                                                            #self.my_tm_tanh.coeffs, dilated or not?
                                                            do_reduction = self._do_reduction,
                                                            )
        print(f"HNRF model is ready in {time() - t0:.2f}seconds")
        #allmodels.append((f"{action}",nrf_evaluator))
        self.featurizer = heaan_nrf.HETreeFeaturizer(h_rf.return_comparator(), self.scheme, self.parms)

    def predict(self, ctx):
        return self.nrf_evaluator(ctx)
    
    
    