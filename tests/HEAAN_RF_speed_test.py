#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import random

from collections import Counter
from time import time
from multiprocessing import Pool


from sklearn.preprocessing import StandardScaler
import fase
from fase.RF.pure_dt import *

def fun(args):
    """
    
    global parameter
    ----------------
    nslots = Number of ciphertext slots, 2**logn
    """
    question, example = args
    feature_name, _, value = question.split(" ")

    n_element = len(example)
    # Always generate an array of length Nslots.
    # And make sure its zero-initialized.
    v1 = np.zeros(nslots)
    v2 = np.zeros(nslots)
    
    v1[:n_element]=float(value)
    v2[:n_element]=np.copy(example[feature_name])
    assert v1.shape == v2.shape
    
    c_crit = he.Ciphertext()
    scheme.encrypt(c_crit, he.Double(v1), n, logp, logq)
    c_val = he.Ciphertext()
    scheme.encrypt(c_val, he.Double(v2), n, logp, logq)

    scheme.subAndEqual(c_crit, c_val)
    # No need to bcootstrap before Sigmoid. We have enough logq left
    csig = he.Ciphertext()
    algo.function(csig, c_crit, "Sigmoid", logp, 4)
    del c_crit, c_val
    return csig


def predict(tree, weights, ww, answers):
    """
    parameters
    ----------
    tree : decision tree as recursive dictionary format
    weights : pre-calculated weights for each question (per example)
    ww : weight to be propagated and multiplied. Set the initial value == 1
    answers : list of (answer, probablity) as answers from ALL leaf nodes
  
    global parameters
    ----------------
    qarr : list of all questions in a tree in np array. 
    """
    question = list(tree.keys())[0]

    i = np.where(question == qarr)[0][0]
    weight = weights[i]

    answer1 = tree[question][0]
    answer2 = tree[question][1]

    w1 = ww * weight
    w2 = ww * (1-weight)

    if not isinstance(answer1, dict):
        answers.append([answer1, w1])
    else:
        predict(answer1, weights, w1, answers)
    if not isinstance(answer2, dict):
        answers.append([answer2,w2])
    else:
        predict(answer2, weights, w2, answers)


def predict_tree(example, tree, answers):

    def recurse(example, node, ww):
        if tree['children_left'][node] >=0 or tree['children_right'][node] >= 0:
            weight = tree['weights'][i]
            
            w1 = ww * weight
            w2 = ww * (1-weight)
            
            recurse(example, tree['children_left'][node])
            recurse(example, tree['children_right'][node])
        else:
            # value == Number of examples that fell in each class during the training phase.
            
            proba = tree['value'][node]
            proba /= np.sum(proba)
            answers.append(proba * ww) # 1D * scalar인 상태 
        
    return recurse(example, 0)


def sum_answers(answers, labels):
    """
    return coded label (or index of label)
    """
    assert isinstance(labels, np.ndarray)
    scores = np.zeros(len(labels))
    
    for aa in answers:
        iscore = np.where(labels == aa[0])[0]
        scores[iscore] += aa[1]
    
    return np.argmax(scores)

def get_most_freq(arr):
    return Counter(arr).most_common(1)[0][0]

def get_questions(tree, qlist):
    """
    Transverse trees to get all questions.
    """
    question = list(tree.keys())[0]
    qlist.append(question)
    
    if isinstance(tree[question][0], dict):
        get_questions(tree[question][0], qlist)
    else:
        pass
        #llist.append(tree[question][0])

    if isinstance(tree[question][1], dict):
        get_questions(tree[question][1], qlist)
    else:
        pass
        #llist.append(tree[question][1])
        

def gen_weight_forest(forest, test_df, ncpu=16):
    """
    Calculate weights for all questions for all examples for all trees.
    
    
    parameters
    ----------
    forest : group of trained trees
    test_df : test dataset in dataframe
    ncput : number of cores for multiprocessing
    """
    ans_forest = []
    for dt in forest:
        qlist=[]
        llist=[]
        get_questions(dt, qlist)

        example = test_df.iloc[:n].to_records()
        pool = Pool(min([len(qlist), ncpu]))
        answers = pool.map(fun,[(qq, example) for qq in qlist])

        ans_forest.append(answers)
    
    return ans_forest


def transform_label(value):
    if value <= 5:
        return "bad"
    else:
        return "good"


import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpga", help="use HEAAN fpga", action="store_true")
    args = parser.parse_args()

    # Load different versions of HEAAN depending on commandline argument
    if args.fpga:
        from fase import HEAAN_fpga as he
    else:
        import fase.HEAAN as he

    random.seed(0)
    
    # Load small test dataset
    df = pd.read_csv("./winequality-red.csv")
    df["label"] = df.quality
    df = df.drop("quality", axis=1)
    label_map = {'good':0, 'bad':1}

    column_names = []
    for column in df.columns:
        name = column.replace(" ", "_")
        column_names.append(name)
    df.columns = column_names

    wine_quality = df.label.value_counts(normalize=True)
    wine_quality = wine_quality.sort_index()

    df["label"] = df.label.apply(transform_label)

    # Standardize
    df.iloc[:,0:-1] = df.iloc[:,0:-1].apply(lambda x: 1. * (x-x.mean())/ x.std(), axis=0)


    # Pure Python RF
    train_df, test_df = train_test_split(df, test_size=0.20)

    n_trees = 5
    dt_max_depth = 6

    forest = random_forest_algorithm(train_df,
                                     n_trees=n_trees,
                                     n_bootstrap=800,
                                     n_features=2,
                                     dt_max_depth=dt_max_depth)
    predictions = random_forest_predictions(test_df, forest)
    accuracy = calculate_accuracy(predictions, test_df.label)
    print("Plain Random Forest model is ready")
    print("Accuracy = {}".format(accuracy))



    # HEAAN context setup
    logp = 30 #
    logq = logp+120 # Number of quantized bits.  Larger logq consumes more memory
    logn = 5 # number of slots... 2^10 = 1024

    n = 1*2**logn
    nslots = n

    parms = {'n':n, 
             'logp':logp, 
             'logq':logq} # for ciphertext constructor

    ring = he.Ring()
    secretKey = he.SecretKey(ring)
    scheme = he.Scheme(secretKey, ring)
    scheme.addBootKey(secretKey, logn, logq + 4)

    algo = he.SchemeAlgo(scheme)


    # ncpu = 1
    if args.fpga:
        print("Evaluating FHE RF using 1 FPGA")
    else:
        print("Evaluating FHE RF using 1 CPU ... will take a while")

    label_coded = np.array([ label_map[ee] for ee in test_df['label'] ])
    t0 = time()
    ans_forest = gen_weight_forest(forest, test_df, ncpu=1)
    print(f"Took {time() - t0:.2f} seconds")


    if not args.fpga:
        print("Evaluating FHE RF using 8 CPU")
        # 8 core
        label_coded = np.array([ label_map[ee] for ee in test_df['label'] ])
        t0 = time()
        ans_forest = gen_weight_forest(forest, test_df, ncpu=8)
        print(f"Took {time() - t0:.2f} seconds")


    # RF results
    dec_ans_forest = np.zeros((len(test_df), len(forest)), dtype=int)
    labels = np.array(list(label_map.keys())) # pass as ndarray
    for j, dt in enumerate(forest):
        acc = []
        qlist=[]
        get_questions(dt, qlist)
        
        dec_ans = []
        for ans in ans_forest[j]:
            prob = scheme.decrypt(secretKey, ans)
            del ans
            arr = np.zeros(n, dtype=np.complex128)
            prob.__getarr__(arr)
            dec_ans.append(arr.real)
            
        all_weights = np.stack(dec_ans, axis=1)
        
        
        for i, ex in enumerate(test_df.iloc[:n].to_records()):
            label = ex[-1]
            qarr = np.array(qlist)

            weighted_answers=[]
            weights = all_weights[i]
            predict(dt, weights, 1., weighted_answers)

            summed_answer = sum_answers(weighted_answers, labels)

            acc.append(summed_answer == label)
            dec_ans_forest[i,j] = summed_answer
        #print(np.sum(acc)/len(acc))

    final_answer = np.apply_along_axis(get_most_freq, 1, dec_ans_forest)
    # AccuracyL logq = 220
    print("FHE RF accuracy:", np.sum(final_answer == label_coded) / len(final_answer))

