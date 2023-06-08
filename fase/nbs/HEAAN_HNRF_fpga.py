import pickle
from time import time
import fase
fase.USE_FPGA = True
import numpy as np
from fase.core.heaan import he

#import fase.core.heaan as he

print("-------")
#from fase import HEAAN as he
from fase import hnrf as hnrf

from fase.hnrf.tree import NeuralTreeMaker
from fase.hnrf import heaan_nrf 
#from fase.hnrf.hetree_nrf import HomomorphicModel 
import torch


def decrypt(secretKey, enc):
    featurized = scheme.decrypt(secretKey, enc)
    arr = np.zeros(n, dtype=np.complex128)
    featurized.__getarr__(arr)
    return arr.real


action = 13
model_dir = '/home/etri_ai2/work/Kinect_BBS_demo/nbs/' #"/home/hoseung/Work/fhenrf/pose/"

small = True

if small:
    fn_model_out = "trained_model13_s.pickle"
    fn_data_out = "BBS_dataset_13_s.pickle"
else:
    fn_model_out = "trained_model_13.pickle"
    fn_data_out = "BBS_dataset_13.pickle"

fn_model = model_dir + fn_model_out
fn_dat = model_dir + fn_data_out

rf_model = pickle.load(open(fn_model, "rb"))

print("model's depth:", rf_model.max_depth)
print("model's tree count:", rf_model.n_estimators)


Nmodel = pickle.load(open(f"trained_NRF_{action}.pickle", "rb"))

dataset = pickle.load(open(fn_dat, "rb"))

X_train = dataset["train_x"]
y_train = dataset["train_y"]
X_valid = dataset["valid_x"]
y_valid = dataset["valid_y"]

print("min max of input dataset")
print(X_train.min(), X_train.max())
print(X_valid.min(), X_valid.max())

#from sklearn.tree import BaseDecisionTree
#from fase.hnrf.tree import NeuralRF

dilatation_factor = 10
polynomial_degree = 10

my_tm_tanh = NeuralTreeMaker(torch.tanh, 
                            use_polynomial=True,
                            dilatation_factor=dilatation_factor, 
                            polynomial_degree=polynomial_degree)


class Param():
    def __init__(self, n=None, logn=None, logp=None, logq=None, logQboot=None):
        self.n = n
        self.logn = logn
        self.logp = logp
        self.logq = logq 
        self.logQboot = logQboot
        if self.logn == None:
            self.logn = int(np.log2(n))


t0 = time()

logq = 540
logp = 30
logn = 14
n = 1*2**logn
slots = n

parms = Param(n=n, logp=logp, logq=logq)

do_reduction=False


ring = he.Ring()
secretKey = he.SecretKey(ring)
scheme = he.Scheme(secretKey, ring, False)

algo = he.SchemeAlgo(scheme)


# reduction때는 right rotation N_class개 필요. 
if do_reduction:
    Nclass = Nmodel.head.shape[0]
    scheme.addLeftRotKeys(secretKey)
    for i in range(Nclass):
        scheme.addRightRotKey(secretKey, i+1) # 
else:
    # reduction 안 하면 하나짜리 rotation만 여러번 반복.
    scheme.addLeftRotKey(secretKey, 1)

from fase.hnrf.hetree import HNRF

all_models =[]

print(f"EVALUATOR ready in {time()-t0:.2f}")

#for i in range(13,14):
h_rf = HNRF(Nmodel)

nrf_evaluator = heaan_nrf.HomomorphicTreeEvaluator.from_model(h_rf,
                                                    scheme,
                                                    parms,
                                                    my_tm_tanh.coeffs,
                                                    do_reduction = do_reduction
                                                    )

featurizer = heaan_nrf.HomomorphicTreeFeaturizer(h_rf.return_comparator(), scheme, parms)
#all_models.append({f"evaulator_{i}":nrf_evaluator,
#                    f"featurizer_{i}":featurizer})

print(f"generating 14 models took {time() - t0:.2f}")


#nrf_evaluator = all_models[0]['evaluator_13']
#featurizer = all_models[0]['featurizer']

for xx, yy in zip(X_valid[:5], y_valid[:5]):
    ctx = featurizer.encrypt(xx)
    t0 = time()
    result = nrf_evaluator(ctx)
    print(f"Took {time() - t0:.2f} seconds")

    pred = []
    for res in result:
        dec = decrypt(secretKey, res)
        pred.append(np.sum(dec))

    print(f"Prediction: {np.argmax(pred)} == {yy}?")