import pickle
from fase.hnrf.tumor import HNRF_builder
import numpy as np
from time import time

def decrypt(context, enc):
    featurized = context.scheme.decrypt(context.secretKey, enc)
    arr = np.zeros(context.parms.n, dtype=np.complex128)
    featurized.__getarr__(arr)
    return arr.real

if __name__=="__main__":
    # Load Trained NeuralRF model
    Nmodel = pickle.load(open("trained_Nmodel_RF.pickle", "rb"))

    # Load dataset
    dd = pickle.load(open("tumor_train.pickle", "rb"))
    X_train, y_train = dd['train_x'], dd['train_y']

    dd = pickle.load(open("tumor_valid.pickle", "rb"))
    X_valid, y_valid = dd['valid_x'], dd['valid_y']


    # Homomorphic NRF
    tumor_hnrf = HNRF_builder(Nmodel, device="cpu")

    # Test accuracy
    for xx, yy in zip(X_valid[:20], y_valid[:20]):
        ctx = tumor_hnrf.featurizer.encrypt(xx)
        t0 = time()
        result = tumor_hnrf.predict(ctx)
        print(f"Took {time() - t0:.2f} seconds")

        pred = []
        for res in result:
            dec = decrypt(tumor_hnrf, res)
            pred.append(np.sum(dec))

        print(f"Prediction: {np.argmax(pred)} == {yy}?")