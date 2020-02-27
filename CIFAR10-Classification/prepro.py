import numpy as np
import pickle
from funcs import load_batch

def normlise(x):
    _min = np.min(x)
    _max = np.max(x)
    return ((x - _min) / (_max - _min))

def one_hor_enc(x):
    enc = np.zeros((len(x),10))

    for i, j in enumerate(x):
        enc[i][j] = 1 
    return enc
    
def prepro(FEATURES, labels, filename):
    features = normalise(features)
    labels = one_hot_enc(labels)

    pickle.dump((features,labels), open(filename, 'wb'))

def Main():
    batches = 5
    valid_features = [] 
    valid_labels = [] 

    for batch_no in range(1, batches+1):
         features, labels = load_batch("cifar-10-batches-py", batch_no)

         index_of_Val = len(features)//10

         prepro(fetures[:-index_of_Val], labels[:-index_of_Val], "prepro_batch" + str(batch_no) +".p")
         
         valid_features.extend(features[-index_of_Val:])
         valid_labels.extend(features[-index_of_Val:])
         
         prepro(np.array(valid_features), np.array(valid_labels), "prepro_Val.p")



         
