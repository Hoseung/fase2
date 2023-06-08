#from typing import Callable, List
import torch
import torch.nn as nn
import numpy as np

def gen_l1(dt):
    feature = dt.tree_.feature
    threshold = dt.tree_.threshold
    
    branch_nodes = np.where(feature >= 0)[0]

    weight = np.zeros((len(branch_nodes), dt.n_features_in_))
    weight[np.arange(len(branch_nodes)), feature[branch_nodes]] = 1

    bias = -threshold[branch_nodes]

    linear = nn.Linear(*weight.shape[::-1])
    linear.weight.data = nn.Parameter(torch.tensor(weight).float())
    linear.bias.data = nn.Parameter(torch.tensor(bias).float())
    return linear


def gen_l2(dt, activation="tanh", eps=0.5):
    feature = dt.tree_.feature

    branch_nodes = list(np.where(feature >= 0)[0])
    leaf_nodes = list(np.where(feature < 0)[0])

    parents={}
    for i in branch_nodes:
        parents[dt.tree_.children_left[i]] = (i, 1)
        parents[dt.tree_.children_right[i]] = (i, -1)

    # node to node index
    n_nodes = dt.tree_.node_count
    ii_b = np.zeros(n_nodes, dtype=int)
    ii_b[branch_nodes] = np.arange(len(branch_nodes))

    ii_l = np.zeros(n_nodes, dtype=int)
    ii_l[leaf_nodes] = np.arange(len(leaf_nodes))

    # generate path map
    path_map = np.zeros((len(leaf_nodes), len(branch_nodes)))
    depths= np.zeros(len(leaf_nodes))
    for leaf in leaf_nodes:
        sib = leaf
        depth = 0
        while sib != 0:
            parent = parents[sib]
            path_map[ii_l[leaf], ii_b[parent[0]]] = parent[1]
            sib = parents[sib][0]
            depth +=1
        depths[ii_l[leaf]] = depth

    # bias = sum( 1*(N_left) -1 * (N_right) ) - depth 
    if activation == "sigmoid":
        bias = (np.sum(path_map, axis=1) - depths)/2 + 1/2
        norm = len(branch_nodes)
    elif activation == "tanh":
        bias = -1*(depths - eps)
        norm = 2*len(branch_nodes)

    linear = nn.Linear(*path_map.shape[::-1])
    linear.weight.data = nn.Parameter(torch.tensor(-1*path_map).float() / norm)
    linear.bias.data   = nn.Parameter(torch.tensor(bias).float() / norm)
    
    return linear

def gen_classifier_head(tree, activation="tanh"):
    feature = tree.tree_.feature
    leaves = list(np.where(feature < 0)[0])

    if activation == "tanh":
        leaf_values = tree.tree_.value[leaves]
        leaf_values = torch.tensor(leaf_values).float()
        leaf_values = leaf_values.squeeze(1) / tree.tree_.value[0].max()

        # We divide by 2 because we have -1 and 1 bits
        bias = leaf_values.sum(dim=0) / 2
        leaf_values = leaf_values / 2
    elif activation == 'sigmoid':
        values = tree.tree_.value[[0] + leaves] # 0은 왜 필요? 

        values = torch.tensor(values).float()
        values = values.squeeze(1)

        root_values = values[0]
        leaf_values = values[1:]

        leaf_values = (leaf_values - root_values.unsqueeze(0)) / root_values.max()
        root_values = root_values / root_values.max()

    head = nn.Linear(*leaf_values.shape)
    head.weight.data = leaf_values.T
    if activation == "tanh":
        head.bias.data = bias
    elif activation == "sigmoid":
        head.bias.data = root_values
    
    return head
    
class NeuralDT(nn.Module):
    """Create a Neural Decision Tree from a SKlearn Decision Tree"""
    def __init__(self, tree,
                 activation,
                 activation_name):
        super().__init__()
        self.activation_name = activation_name
        self.comparator = gen_l1(tree)
        self.matcher = gen_l2(tree, activation=activation_name)

        self.head = gen_classifier_head(tree, activation=activation_name)

    def forward(self,x):
        comparisons = self.comparator(x)
        comparisons = self.activation(comparisons)

        matches = self.matcher(comparisons)
        matches = self.activation(matches)

        output = self.head(matches)

        return output

    def return_weights(self):
        """Returns the weights used in each layer."""
        w0 = self.comparator.weight.data.numpy()
        b0 = self.comparator.bias.data.numpy()

        w1 = self.matcher.weight.data.numpy()
        b1 = self.matcher.bias.data.numpy()

        w2 = self.head.weight.data.numpy()
        b2 = self.head.bias.data.numpy()

        return w0, b0, w1, b1, w2, b2