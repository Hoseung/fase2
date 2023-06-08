import numpy as np
from typing import List

def extract_diagonals(matrix: np.ndarray) -> List[np.ndarray]:
    """Extracts the diagonals of the matrix"""
    assert matrix.shape[0] == matrix.shape[1], "Non square matrix"
    dim = matrix.shape[0]

    diagonals = []

    for i in range(dim):
        diagonal = []
        for j in range(dim):
            diagonal.append(matrix[j][(j+i) % dim])
        diagonal = np.array(diagonal)
        diagonals.append(diagonal)
    return diagonals

def pad_along_axis(array: np.ndarray, target_length, axis=0):
    """Pads an array to a given target length, on a given axis."""
    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)

    if pad_size <= 0:
        return array

    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)

    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    return b
