__all__ = ['NeuralTreeMaker', 
           'DEFAULT_POLYNOMIAL_DEGREE', 'DEFAULT_DILATATION_FACTOR',
           'DEFAULT_BOUND', 
           'pad_tensor', 'pad_neural_tree', 'NeuralRF', 'CrossEntropyLabelSmoothing']

# Cell
import numpy as np

from typing import List, Callable
import torch
import torch.nn as nn
#import torch.nn.functional as F

from sklearn.tree import BaseDecisionTree

from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial import Polynomial
from .defaults import DEFAULT_POLYNOMIAL_DEGREE, DEFAULT_DILATATION_FACTOR, DEFAULT_BOUND

from .nrf import NeuralDT

class NeuralTreeMaker:
    """Base class to """
    def __init__(self,
                 activation: Callable,
                 dilatation_factor : float = DEFAULT_DILATATION_FACTOR,
                 use_polynomial : bool = False,
                 polynomial_degree : int = DEFAULT_POLYNOMIAL_DEGREE, bound: float = DEFAULT_BOUND):

        # first we need to define the activation used
        activation_fn = lambda x: activation(x * dilatation_factor)
        self.activation_name = activation.__name__
        if use_polynomial:
            domain = [-bound, bound]
            activation_fn_numpy = lambda x: activation_fn(torch.tensor(x))
            self.activation = Chebyshev.interpolate(activation_fn_numpy,deg=polynomial_degree,domain=domain)
            self.coeffs = Polynomial.cast(self.activation).coef
        else:
            self.activation = activation_fn
            self.coeffs = None

    def make_tree(self, tree: BaseDecisionTree):
        neural_tree = NeuralDT(tree,
                                         self.activation, 
                                         self.activation_name, 
                                         )
        return neural_tree


def check_output_range(m, i, o, threshold=1):
    rows_outside_range = ((torch.abs(o) > threshold).float().sum(dim=1) > 0).numpy()
    idx_outside_range = np.arange(len(rows_outside_range))[rows_outside_range]

    assert len(idx_outside_range) == 0, f"""Out of range outputs for module {m}: \n
    {idx_outside_range} \n
    Rows with outside range : \n
    {o.numpy()[idx_outside_range]}"""


# Cell
def pad_tensor(tensor, target, dim=0, value=0):
    # If the tensor is already at the target size we return it
    if tensor.shape[dim] >= target:
        return tensor
    else:
        shape = list(tensor.shape)
        shape[dim] = target - tensor.shape[dim]

        padding = torch.ones(shape) * value
        output = torch.cat([tensor,padding], dim=dim)
        return output

def pad_neural_tree(neural_tree, n_nodes_max, n_leaves_max):
    w0, b0 = neural_tree.comparator.weight.data.clone(), neural_tree.comparator.bias.data.clone()

    # First we pad the output size of the comparator
    neural_tree.comparator = nn.Linear(w0.shape[1], n_nodes_max)
    neural_tree.comparator.weight.data = pad_tensor(w0, n_nodes_max, dim=0)
    neural_tree.comparator.bias.data = pad_tensor(b0, n_nodes_max, dim=0)

    w1, b1 = neural_tree.matcher.weight.data.clone(), neural_tree.matcher.bias.data.clone()
    # Then we pad the output and the input size of the matcher
    neural_tree.matcher = nn.Linear(n_nodes_max, n_leaves_max)
    neural_tree.matcher.weight.data = pad_tensor(pad_tensor(w1, n_nodes_max, dim=1), n_leaves_max, dim=0)
    neural_tree.matcher.bias.data = pad_tensor(b1, n_leaves_max, dim=0)

    w2, b2 = neural_tree.head.weight.data.clone(), neural_tree.head.bias.data.clone()
    neural_tree.head = nn.Linear(n_leaves_max, w2.shape[0])
    neural_tree.head.weight.data = pad_tensor(w2, n_leaves_max, dim =1)
    neural_tree.head.bias.data = b2

# Cell
class NeuralRF(nn.Module):
    def __init__(self, trees: List[BaseDecisionTree],
                 tree_maker: NeuralTreeMaker,
                 weights: torch.Tensor = None, trainable_weights:bool = False,
                 bias: torch.Tensor = None, trainable_bias:bool = False):

        super(NeuralRF, self).__init__()

        self.n_trees = len(trees)
        self.activation = tree_maker.activation

        # First we need to create the neural trees
        neural_trees = []
        n_nodes = []
        n_leaves = []
        for tree in trees:
            neural_tree = tree_maker.make_tree(tree)
            n_nodes.append(neural_tree.comparator.weight.data.shape[0])
            n_leaves.append(neural_tree.matcher.weight.data.shape[0])
            neural_trees.append(neural_tree)

        # Then we pad our neural trees according to the biggest tree in the forest
        n_nodes_max = max(n_nodes)
        n_leaves_max = max(n_leaves)

        self.n_leaves_max = n_leaves_max

        for neural_tree in neural_trees:
            pad_neural_tree(neural_tree, n_nodes_max, n_leaves_max)

        self.neural_trees = neural_trees

        # Then we create the parameters for the Neural Random Forest
        comparators = [neural_tree.comparator.weight.data.unsqueeze(-1) for neural_tree in neural_trees]
        comparator = torch.cat(comparators, dim=-1)
        comparator = comparator.permute(1,0,2)
        comparator = nn.Parameter(comparator)
        self.register_parameter("comparator", comparator)

        comparator_bias = [neural_tree.comparator.bias.data.unsqueeze(-1) for neural_tree in neural_trees]
        comparator_bias = torch.cat(comparator_bias, dim=-1)
        comparator_bias = nn.Parameter(comparator_bias)
        self.register_parameter("comparator_bias", comparator_bias)

        matchers = [neural_tree.matcher.weight.data.unsqueeze(-1) for neural_tree in neural_trees]
        matcher = torch.cat(matchers, dim=-1)
        matcher = nn.Parameter(matcher)
        self.register_parameter("matcher", matcher)

        matcher_bias = [neural_tree.matcher.bias.data.unsqueeze(-1) for neural_tree in neural_trees]
        matcher_bias = torch.cat(matcher_bias, dim=-1)
        matcher_bias = nn.Parameter(matcher_bias)
        self.register_parameter("matcher_bias",matcher_bias)

        heads = [neural_tree.head.weight.data.unsqueeze(-1) for neural_tree in neural_trees]
        head = torch.cat(heads, dim=-1)
        head = nn.Parameter(head)
        self.register_parameter("head", head)

        head_bias = [neural_tree.head.bias.data.unsqueeze(-1) for neural_tree in neural_trees]
        head_bias = torch.cat(head_bias, dim=-1)
        head_bias = nn.Parameter(head_bias)
        self.register_parameter("head_bias", head_bias)

        if not torch.is_tensor(weights):
            weights = torch.ones(self.n_trees) * (1. / self.n_trees)

        if trainable_weights:
            weights = nn.Parameter(weights)
            self.register_parameter("weights", weights)
        else:
            self.register_buffer("weights", weights)

        if not torch.is_tensor(bias):
            c = neural_tree.head.weight.data.shape[0]
            bias = torch.zeros(c)

        if trainable_bias:
            bias = nn.Parameter(bias)
            self.register_parameter("bias",bias)
        else:
            self.register_buffer("bias",bias)

    def forward(self, x):
        comparisons = self.compare(x)
        #print("1. comparisons", comparisons)
        matches = self.match(comparisons)
        #print("2. match", matches)
        outputs = self.decide(matches)
        #print("3. outputs", outputs)

        return outputs

    def compare(self, x):
        """
        https://rockt.github.io/2018/04/30/einsum
        """
        # x.shape = (N_example, N_feature) => kj
        # comparator.shape = (N_feature, N_branches, N_trees) => jil
        # result be kil, or (N_example, N_branches, N_Trees)
        comparisons = torch.einsum("kj,jil->kil",x,self.comparator) + self.comparator_bias.unsqueeze(0)
        comparisons = self.activation(comparisons)
        return comparisons

    def match(self, comparisons):
        matches = torch.einsum("kjl,ijl->kil",comparisons, self.matcher) + self.matcher_bias
        matches = self.activation(matches)
        return matches

    def decide(self, matches):
        outputs = torch.einsum("kjl,cjl->kcl",matches,self.head) + self.head_bias
        outputs = (outputs * self.weights.expand_as(outputs)).sum(dim=-1)
        outputs = outputs + self.bias.expand_as(outputs)
        return outputs

    def get_weight_and_bias(self, module:str):
        weight = getattr(self, module)
        bias = getattr(self, module + "_bias")

        return weight, bias

    def freeze_layer(self, module: str):
        weight, bias = self.get_weight_and_bias(module)
        weight.requires_grad = False
        bias.requires_grad = False

    def unfreeze_layer(self, module: str):
        weight, bias = self.get_weight_and_bias(module)
        weight.requires_grad = True
        bias.requires_grad = True

    def return_weights(self):
        W0 = list(self.comparator.data.permute(2,1,0).numpy())
        B0 = list(self.comparator_bias.data.permute(1,0).numpy())

        W1 = list(self.matcher.data.permute(2,0,1).numpy())
        B1 = list(self.matcher_bias.data.permute(1,0).numpy())

        W2 = list(self.head.data.permute(2,0,1).numpy())
        B2 = list(self.head_bias.data.permute(1,0).numpy())

        return W0, B0, W1, B1, W2, B2

    def to_device(self, device):
        self.comparator.data = self.comparator.data.to(device)
        self.comparator_bias.data = self.comparator_bias.data.to(device)

        self.matcher.data = self.matcher.data.to(device)
        self.matcher_bias.data = self.matcher_bias.data.to(device)

        self.head.data = self.head.data.to(device)
        self.head_bias.data = self.head_bias.data.to(device)


class CrossEntropyLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, dim=-1, reshape=True):
        """https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06"""
        
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim
        self.reshape = reshape

    def forward(self, pred, target):
        K = pred.shape[-1]

        if self.reshape:
            pred = pred.view(-1,K)
            target = target.view(-1)

        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (K - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
