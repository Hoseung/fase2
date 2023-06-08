import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler#StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def letter2int(letter):
    """Tumor dataset-specific function.
    """
    out = ord(letter) - 65.
    if out > 25:
        out -= 32.
    return out

def chromosome2num(obj):
    """Tumor dataset-specific function.
    """
    if obj == 'X':
        return 23.
    elif obj == 'Y':
        return 24.
    else:
        return float(obj)    

def extract_sk_dt(tree):
    """Extract most relevant information from SKLearn Decision Tree

    Parameters
    ----------
    tree : SKlearn Decision Tree instance

    Returns
    -------
    dict
        Decision Tree Dict. A dict of essential tree parameters.
    """
    tree_ = tree.tree_
    dt = {'children_left':tree_.children_left,
          'children_right': tree_.children_right,
          'feature':tree_.feature,
          'threshold':tree_.threshold,
          'value':tree_.value,
          'classes':tree.classes_,
          'branch':np.where(tree_.feature >= 0)[0],
          'leaf':np.where(tree_.feature < 0)[0],
          'depth':tree.max_depth}

    return dt

class Tumordata:
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

class TumorRF(Tumordata):    
    def __init__(self, data_dir):
        """Load and preprocess the data
        """
        self.data_dir = data_dir
        self.load_all_files()
        self.le = LabelEncoder()
        
        self.preprocess_df()
        self.standardize()
        print("Preprocessing done and train data are ready")

    def load_all_files(self, data_dir=None):
        """Load tumor data. (Specific data structure expected.)
        """
        filenames = self.filenames
        variants_ext = self.variants_ext
        col_Names = self.col_Names
        
        if data_dir is None:
            data_dir = self.data_dir
            
        df0 = pd.DataFrame(columns=col_Names)

        for filename in filenames:
            dataframe = pd.read_csv(data_dir + filename+variants_ext, sep='\t', names = col_Names)
            dataframe['tumorLOC'] = filename
            df0 = pd.concat([df0, dataframe], axis = 0)

        del dataframe
        self.df0 = df0        
        
    def preprocess_df(self):
        """preprocessing tumor data.

        NOTE
        ----
        Significant performance improvement can be achieved if 
        """
        df0 = self.df0
        le = self.le
        
        # fill missing values.
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

        df["geneNameNum"]   = geneNameNum
        df["chromosomeNum"] = chromosomeNum
        df["location"]      = le.fit_transform(df["location"]).astype('float')

        df["mutationType"]  = le.fit_transform(df["mutationType"]).astype('float')
        df["SNP?"]          = le.fit_transform(df["SNP?"]).astype('float')
        df["mutationsEffect_1"] = le.fit_transform(df["mutationsEffect_1"]).astype('float')
        df["mutationsEffect_2"] = le.fit_transform(df["mutationsEffect_2"]).astype('float')
        self.data_label = np.array(le.fit_transform(df["tumorLOC"]).tolist())

        del df["geneName"]
        del df["chromosome"]
        del df['location']

        data = df.copy()
        del data['tumorLOC']

        self.data = data
        
        
    def standardize(self):
        """standardize data into strict range [-1,1] so that approximate sigmoid/tanh behave nice.
        """
        sc = MinMaxScaler(feature_range=[0,1])
        self.data = sc.fit_transform(self.data)
        

    def split_data(self, fn_dump=None, test_size=0.2):
        """Split the data set into train and valid set.

        parameters
        ----------
        fn_dump: (optional) str
            prefix for dataset dump

        test_size: float (0,1)
            fraction of test dataset to whole data set
        """
        X_train, X_valid, Y_train, Y_valid = \
            sklearn.model_selection.train_test_split(self.data, 
                                                     self.data_label, 
                                                     test_size=test_size, 
                                                     stratify=self.data_label)
        if fn_dump is not None:
            fn_train = fn_dump+"train.pickle"
            fn_valid = fn_dump+"valid.pickle"
            pickle.dump({"train_x":X_train,
                         "train_y":Y_train, 
                         "label_encoder":self.le}, open(fn_train, "wb"))
            print("Saved Training set as", fn_train)
            pickle.dump({"valid_x":X_valid,
                         "valid_y":Y_valid, 
                         "label_encoder":self.le}, open(fn_valid, "wb"))
            print("Saved Validation set as", fn_valid)

        return X_train, X_valid, Y_train, Y_valid
            
        
    def train(self, X_train, Y_train, fn_out="trained_RF.pickle", ntree=21, depth=8):
        """Construct and train a Random Forest model
        """
        print(f"Training a Random Forest with {ntree} trees of depth {depth}.")
        
        self.rf = RandomForestClassifier(n_estimators=ntree, max_depth = depth)
        self.rf.fit(X_train, Y_train)
        self.save(fn_out)
        self.acc_train = self.rf.score(X_train, Y_train)
        print(f"Train Done: train accuracy = {100*self.acc_train:.3f}%")

    def predict(self, X_valid):
        return self.rf.predict(X_valid)
    
    def accuracy(self, Y_valid, y_preds):
        self.acc = accuracy_score(Y_valid, y_preds)
        print(f"Test accuracy {100*self.acc:.3f}%")

    def save(self, fn_out='trained_RF.pickle'):
        """Deprecated version.
        """
        RF = [extract_sk_dt(dt) for dt in self.rf]
        pickle.dump(RF, open(fn_out, "wb"))
        print(f"Trained model saved at: {fn_out}")

    def save2(self, fn_out="trained_RF.pickle"):
        self.data = None
        pickle.dump(self, open(fn_out, "wb"))
        print(f"Trained model saved at: {fn_out}")

    def save_model(self, fn):
        pickle.dump(self.rf, open(fn, "wb"))