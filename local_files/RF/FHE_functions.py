# 1. np.copy is necessary. 
# Recarr's view method doesn't directly represent the shape of memory. 
# C++ gets the head address of the array the next element in memory isn't the one 
# next in the view().
# 2. Match the shape of c_crit and c_val. 

from multiprocessing import Pool
import numpy as np
from fase import heaan_loader
he = heaan_loader.load()
from collections import Counter

def decrypt_all_weights(ans_forest, sch):
    """decrypt weights and reshape 
    """
    dec_each_tree = []
    for ans_each_tree in ans_forest:
        dec_each_node = []
        for ans_each_node in ans_each_tree: # loop over each node
            dec_each_node.append(sch.decrypt(ans_each_node))
            del ans_each_node
        dec_each_tree.append(dec_each_node)
        
    all_trees_nodes_per_example = [np.stack(a, axis=1) for a in dec_each_tree]
    return all_trees_nodes_per_example

def predict_tree(tree):

    def recurse(node, ww):
        if tree['children_left'][node] >=0 or tree['children_right'][node] >= 0:
            weight = tree['prb'][node] # branching prb of this node
            
            w1 = ww * weight
            w2 = ww * (1-weight)
            
            recurse(tree['children_left'][node], w1)
            recurse(tree['children_right'][node], w2)
        else:
            # value == Number of examples that fell in each class during the training phase.
            proba = tree['value'][node]
            proba /= np.sum(proba) # Normalize
            tree['answer'] += proba.squeeze() * ww # normalized prediction * leaf prob
         
    recurse(0, 1)# start from node 0 with an initial weight 1.


def replicate_tree_with_weight(dt, weights_tree_node_ex):
    tt={}
    tt['prb'] = np.zeros(len(dt['feature']))
    tt['prb'][dt['branch']] = weights_tree_node_ex
    tt['children_left'] = dt['children_left']
    tt['children_right'] = dt['children_right']
    tt['value'] = dt['value']
    tt['classes'] = dt['classes']
    tt['answer'] = np.zeros(len(tt['classes']))
    return tt

def cipher_sigmoid(ctxt, c_parms, degree=4):
    c_sig = he.Ciphertext(**c_parms)
    algo.function(c_sig, ctxt, "Sigmoid", c_parms["logp"], degree) # Keyword string must be precise. 
    return c_sig

def is_smaller_ab(self, val, crit, parms):
    c_crit = self.encrypt(crit, parms)
    
    c_val = self.encrypt(val, parms)
    self.scheme.subAndEqual(c_crit, c_val)

    is_smaller = self.cipher_sigmoid(c_crit, parms, degree=5) 
    
    return is_smaller
