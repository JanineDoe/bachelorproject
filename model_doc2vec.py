import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing

from helper_functions import confusion_matrix, accuracy, kmer_encoder
from gensim.models.callbacks import CallbackAny2Vec

np.random.seed(7)

# parameters
seq_length = 1273
number_proteins = 20
voc_size = number_proteins + 1
number_clades = 17
embedding_dim = 32  # kinda random


# shamelessly robbed callback function
class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 1

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


def create_d2v_model(vector_size, epochs, kmers, mini_sample: 0):
    kmer_size = kmers
    num_epochs = epochs
    vec_size = vector_size

    # load training and test data
    if mini_sample == 1:
        training_data = pd.read_csv(
            'C:/Users/Anonyma/Desktop/Unikram/Bachelorprojekt/data_bachelorproject/training_data_mini_sample.csv')
        test_data = pd.read_csv(
            'C:/Users/Anonyma/Desktop/Unikram/Bachelorprojekt/data_bachelorproject/test_data_mini_sample.csv')
    else:
        training_data = pd.read_csv(
            'C:/Users/Anonyma/Desktop/Unikram/Bachelorprojekt/data_bachelorproject/training_data.csv')
        test_data = pd.read_csv('C:/Users/Anonyma/Desktop/Unikram/Bachelorprojekt/data_bachelorproject/test_data.csv')

    X_test = test_data['Sequence']
    y_test = test_data['Clade']
    X_train = training_data['Sequence']
    y_train = np.array(training_data['Clade'])

    test = []
    for i in range(0, len(y_test)):
        test.append(TaggedDocument(kmer_encoder(X_test[i], kmer_size), [y_test[i]]))

    train = []
    for i in range(0, len(y_train)):
        train.append(TaggedDocument(kmer_encoder(X_train[i], kmer_size), [y_train[i]]))

    ep_log = EpochLogger()

    # create the model
    cores = multiprocessing.cpu_count()
    model = Doc2Vec(vector_size=vec_size, dm=1, window=1, workers=cores, seed=7, epochs=num_epochs)
    model.build_vocab(train)
    model.train(train, total_examples=len(y_train), epochs=num_epochs, callbacks=[ep_log])

    model.save('model_doc2vec.d2v')

    # evaluate model
    '''
        # remove later-----------------------------------------------------------------------------------------------------
    y_pred = []
    for i in range(0, len(y_test)):
        pred_vec = model.infer_vector(kmer_encoder(X_test[i], kmer_size))
        y_pred.append((model.dv.most_similar(positive=pred_vec, topn=1))[0][0])
        print(i + 1, ' of ', len(y_test))

    print(y_pred)
    print("Accuracy: %.10f%%" % (accuracy(y_test, y_pred) * 100))
    for i in range(0, number_clades+1):
       print(confusion_matrix(y_test, y_pred)[i])'''
    return model


model = create_d2v_model(epochs=20, kmers=20, vector_size=10, mini_sample=0)

# ep = 20, k =  20, vec_size = 10  ------------ accurracy: 0,
# ep = 30, k =  20, vec_size = 100  ------------ accurracy: 0,416
# ep = 30, k =  20, vec_size = 200  ------------ accurracy:
# ep = 30, k =  20, vec_size = 300  ------------ accurracy:
# ep = 30, k =  20, vec_size = 500  ------------ accurracy: