import pandas as pd
import multiprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec
from helper_functions import confusion_matrix, accuracy, kmer_encoder


# parameters
seq_length = 1273
number_proteins = 20
voc_size = number_proteins + 1
number_clades = 17
embedding_dim = 32


# shamelessly stolen callback function
class EpochLogger(CallbackAny2Vec):
    """Callback to log information about training"""

    def __init__(self):
        self.epoch = 1

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


def create_d2v_model(vector_size, number_epochs, kmer_size, eval=0):
    """creates and saves a trained doc2vec model. Set eval to true to get a crude evaluation of the model"""
    kmer_size = kmer_size
    num_epochs = number_epochs
    vec_size = vector_size

    # load training and test data
    training_data = pd.read_csv('training_data_mini_sample.csv')
    test_data = pd.read_csv('test_data_mini_sample.csv')

    X_test = test_data['Sequence']
    y_test = test_data['Clade']
    X_train = training_data['Sequence']
    y_train = training_data['Clade']

    # encode sequences of amino acids as kmers of length kmer_size
    test = []
    for i in range(0, len(y_test)):
        test.append(TaggedDocument(kmer_encoder(X_test[i], kmer_size), [y_test[i]]))

    train = []
    for i in range(0, len(y_train)):
        train.append(TaggedDocument(kmer_encoder(X_train[i], kmer_size), [y_train[i]]))

    ep_log = EpochLogger()

    # create, train and save the model
    cores = multiprocessing.cpu_count()
    model = Doc2Vec(vector_size=vec_size, dm=1, window=1, workers=cores, epochs=num_epochs)
    model.build_vocab(train)
    model.train(train, total_examples=len(y_train), epochs=num_epochs, callbacks=[ep_log])

    model.save('model_doc2vec.d2v')

    if eval:
        y_pred = []
        for i in range(0, len(y_test)):
            pred_vec = model.infer_vector(kmer_encoder(X_test[i], kmer_size))
            y_pred.append((model.dv.most_similar(positive=pred_vec, topn=1))[0][0])
        print(y_pred)
        print("Accuracy: %.10f%%" % (accuracy(y_test, y_pred) * 100))
        for i in range(0, number_clades+1):
           print(confusion_matrix(y_test, y_pred)[i])

    return model
