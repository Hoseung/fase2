from fase.RF.sktree import TumorRF
import pickle

if __name__=="__main__":
    data_dir = "/home/hoseung/Dropbox/DeepInsight/2021ETRI/RF_Tumor_Classification/data/"

# Init TumorRf class
# Pre-defined preprocessing steps are taken automatically. 
    tumor_rf = TumorRF(data_dir)

# Split the data and keep validation data separately. 
    X_train, X_valid, Y_train, Y_valid = tumor_rf.split_data(fn_dump="tumor_")

# Train the model
    tumor_rf.train(X_train, Y_train, fn_out="trained_RF.pickle", ntree=21, depth=8)

# Test the model against validation dataset.
    y_preds = tumor_rf.predict(X_valid)

# Caculate validation performance 
    tumor_rf.accuracy(Y_valid, y_preds)

#pickle.dump({"train_x":X_train, "train_y":Y_train}, open("trainset.pickle", "wb"))
    pickle.dump(tumor_rf, open("whole_rf.pickle", "wb"))





