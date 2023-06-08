_all__ = ['to_list_and_duplicate', 'to_list_and_pad', 'HomomorphicModel', 'HEDT',
           'HNRF']

import numpy as np
from .utils import extract_diagonals, pad_along_axis
from .nrf import NeuralDT 
from .tree import NeuralRF

def to_list_and_duplicate(array):
    """Takes an array, append a 0 then copies itself once more.

    This step is necessary to do matrix multiplication with Galois rotations.
    This is used on the bias of the comparator.
    """
    array = list(array)
    array = array + [0] + array
    return array

def to_list_and_pad(array):
    """Takes an array, append len(array) of zeros 

    This step is necessary to do matrix multiplication with Galois rotations.
    This is used on the diagonal vectors.
    """
    array = list(array)
    array = array + [0] * (len(array) - 1)
    return array


class HomomorphicModel:
    """Base class for Homormorphic Decision Trees and Random Forest.

    As the Homomorphic Evaluator will only need weights, and comparator
    for the Homomorphic Featurizer, a model should only return these two.
    """
    def return_weights(self):
        return self.b0, self.w1, self.b1, self.w2, self.b2

    def return_comparator(self) -> np.ndarray:
        """Returns the comparator, which is a numpy array of the comparator,
        with -1 indices for null values. The array is repeated for Galois
        rotations before the multiplication."""

        comparator = list(self.comparator)
        comparator = comparator + [-1] + comparator
        comparator = np.array(comparator)
        return comparator

class HEDT(HomomorphicModel):
    """Homomorphic Decision Tree, which extracts appropriate weights for
    homomorphic operations from a Neural Decision Tree."""

    def __init__(self, w0, b0, w1, b1, w2, b2):
        # We first get the comparator and set to -1 the rows that were padded
        comparator = w0
        padded_rows = (comparator.sum(axis=1) == 0)

        # We then get the indices of non padded rows
        comparator = comparator.argmax(axis=1)
        comparator[padded_rows] = -1
        self.comparator = comparator

        self.n_leaves = w1.shape[0]

        # We add a 0 then copy the initial
        self.b0 = to_list_and_duplicate(b0)

        # For weights, we first pad the columns, then extract the diagonals, and pad them
        w1 = pad_along_axis(w1, w1.shape[0], axis=1)
        w1 = extract_diagonals(w1)
        self.w1 = [to_list_and_pad(w1[i]) for i in range(len(w1))]

        self.b1 = to_list_and_pad(b1)

        self.w2 = [to_list_and_pad(w2[c]) for c in range(len(w2))]

        self.b2 = [to_list_and_pad(([b2[c] / self.n_leaves]) * self.n_leaves) for c in range(len(b2))]

    @classmethod
    def from_neural_tree(cls, neural_tree: NeuralDT):
        return cls(neural_tree.return_weights())

class HNRF(HomomorphicModel):
    """"""
    def __init__(self, neural_rf: NeuralRF, device="cpu"):

        homomorphic_trees = [HEDT(w0, b0, w1, b1, w2, b2)
                             for (w0, b0, w1, b1, w2, b2) in zip(*neural_rf.return_weights())]

        B0, W1, B1, W2, B2 = [], [], [], [], []
        comparator = []

        for h in homomorphic_trees:
            b0, w1, b1, w2, b2 = h.return_weights()
            B0 += b0
            W1.append(w1)
            B1 += b1
            W2.append(w2)
            B2.append(b2)
            comparator += list(h.return_comparator())

        self.comparator = comparator

        W1 = list(np.concatenate(W1, axis=-1))
        W2 = list(np.concatenate(W2, axis=-1))
        B2 = list(np.concatenate(B2, axis=-1))

        # We will multiply each class vector with the corresponding weight for each tree
        weights = neural_rf.weights
        block_size = neural_rf.n_leaves_max * 2 - 1
        weights = [[weight.item()] * block_size for weight in weights]
        weights = np.concatenate(weights)

        W2 = [w2 * weights for w2 in W2]
        B2 = [b2 * weights for b2 in B2]

        self.b0 = B0
        self.w1 = W1
        self.b1 = B1
        self.w2 = W2
        self.b2 = B2
