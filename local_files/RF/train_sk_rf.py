#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import argparse

from fase.RF.utils_pure import extract_sk_dt


def letter2int(letter):
    out = ord(letter) - 65.
    if out > 25:
        out -= 32.
    return out

def chromosome2num(obj):
    if obj == 'X':
        return 23.
    elif obj == 'Y':
        return 24.
    else:
        return float(obj)    

def preprocess_df(df0):
    """preprocessing tumor data
    """
    
    # 결측값 추가
    df0 = df0.fillna(0)
    
    geneNameNum = []
    for letter in df0["geneName"].tolist():
        geneNameNum.append(letter2int(letter[0]))

    chromosomeNum = []
    for letter in df0["chromosome"].tolist():
        chromosomeNum.append(chromosome2num(letter))

    df = df0.copy()
    df["sampleID"]      = le.fit_transform(df["sampleID"]).astype('float')
    df["location"]      = le.fit_transform(df["location"]).astype('float')
    df["geneName"]      = le.fit_transform(df["geneName"]).astype('float')
    ####
    df["geneNameNum"]   = geneNameNum
    df["chromosomeNum"] = chromosomeNum
    df["location"]      = le.fit_transform(df["location"]).astype('float')

    df["mutationType"]  = le.fit_transform(df["mutationType"]).astype('float')
    df["SNP?"]          = le.fit_transform(df["SNP?"]).astype('float')
    df["mutationsEffect_1"] = le.fit_transform(df["mutationsEffect_1"]).astype('float')
    df["mutationsEffect_2"] = le.fit_transform(df["mutationsEffect_2"]).astype('float')
    data_label = np.array(le.fit_transform(df["tumorLOC"]).tolist())

    del df["geneName"]
    del df["chromosome"]
    del df['location']
    
    data = df.copy()
    del data['tumorLOC']
    
    return data, data_label

def train(X_train, Y_train, ntree=21, depth=8):
    random_forest = RandomForestClassifier(n_estimators=ntree, max_depth = depth)
    random_forest.fit(X_train, Y_train)
    print("Train accuracy", random_forest.score(X_train, Y_train))
    
    return random_forest

def load_all_files(data_dir):
    filenames = ["Bladder", 
                 "Breast",
                 "Bronchusandlung", 
                 "Cervixuteri", 
                 "Colon",
                 "Corpusuteri", 
                 "Kidney", 
                 "Liverandintrahepaticbileducts", 
                 "Ovary", 
                 "Skin", 
                 "Stomach"]

    variants_ext = "_challenge_variants.txt"

    col_Names = ["sampleID", "geneName", "chromosome", 
                 "location","mutationType","SNP?",
                 "mutationsEffect_1","mutationsEffect_2"]
    
    df0 = pd.DataFrame(columns=col_Names)

    for filename in filenames:
        dataframe = pd.read_csv(data_dir + filename+variants_ext, sep='\t', names = col_Names)
        dataframe['tumorLOC'] = filename
        df0 = pd.concat([df0, dataframe], axis = 0)

    del dataframe
    return df0


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-dir", "--data_dir", help="data path")
    parser.add_argument("-n", "--ntree", type=int, default=21, help="Number of RF trees")
    parser.add_argument("-d", "--depth", type=int, default=8, help="Depth of RF trees")
    parser.add_argument("-p", "--fn_model", type=str, 
                                default="trained_RF.pickle",
                                help="trained parameter file name")
    parser.add_argument("-o", "--fn_valid", type=str, 
                                default="tumor_testset.pickle", 
                                help="valid datatset file name")

    args = parser.parse_args()

    data_dir = args.data_dir
    ntree = args.ntree
    depth = args.depth
    fn_model = args.fn_model
    fn_valid = args.fn_valid
    

    le = LabelEncoder()
    sc = StandardScaler()
    np.random.seed(123) # 결과 재현이 가능하도록 시드를 설정합니다.
    

    df0 = load_all_files(data_dir)
    
    data, data_label = preprocess_df(df0)
    
    # Standardize
    data = sc.fit_transform(data)
    
    # Split train and valid dataset
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(data, 
                                                                                data_label, 
                                                                                test_size=0.15, 
                                                                                stratify=data_label)

    pickle.dump((X_test, Y_test, le), open(fn_valid, "wb"))
    
    # Train RF model
    random_forest = train(X_train, Y_train, ntree=ntree, depth=depth)
    
    # Test accuracy of the original model 
    y_preds = random_forest.predict(X_test)
    print("Test accuracy", accuracy_score(Y_test, y_preds))
    
    # dump tree questions
    RF = [extract_sk_dt(dt) for dt in random_forest]
    pickle.dump(RF, open(fn_model, "wb"))



