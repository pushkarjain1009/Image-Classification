import pickle

def load_batch(dataset_path, batch_no):
    d = open(dataset_path + '/data_batch_' + str(batch_no), 'rb')
    batch = pickle.load(d, encoding="latin1")

    final= batch['data'].reshape(len(batch['data']),3,32,32).transpose(0,2,3,1)
    labels= batch['labels']

    return final,labels

