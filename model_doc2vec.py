import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
from helper_functions import confusion_matrix, accuracy



np.random.seed(7)


# parameters
seq_length = 1273
number_proteins = 20
voc_size = number_proteins + 1
number_clades = 17
embedding_dim = 32   # kinda random

# load training and test data
#training_data = pd.read_csv('C:/Users/Anonyma/Desktop/Unikram/Bachelorprojekt/data_bachelorproject/training_data_mini_sample.csv')
#test_data = pd.read_csv('C:/Users/Anonyma/Desktop/Unikram/Bachelorprojekt/data_bachelorproject/test_data_mini_sample.csv')
training_data = pd.read_csv('C:/Users/Anonyma/Desktop/Unikram/Bachelorprojekt/data_bachelorproject/training_data.csv')
test_data = pd.read_csv('C:/Users/Anonyma/Desktop/Unikram/Bachelorprojekt/data_bachelorproject/test_data.csv')

X_test = test_data['Sequence']
y_test = test_data['Clade']
X_train = training_data['Sequence']
y_train = np.array(training_data['Clade'])



test = []
for i in range(0, len(y_test)):
    test.append(TaggedDocument(X_test[i], [y_test[i]]))

train = []
for i in range(0, len(y_train)):
    train.append(TaggedDocument(X_train[i], [y_train[i]]))








# create the model
cores = multiprocessing.cpu_count()
model = Doc2Vec(vector_size=300, dm=1, window=2, workers=cores, seed=7, epochs=30)
model.build_vocab(train)
model.train(train, total_examples=len(y_train), epochs=30)




#model.save('./model_doc2vec.d2v')

# evaluate model

y_pred = []
for i in range(0, len(y_test)):
    pred_vec = model.infer_vector([X_test[i]])
    y_pred.append((model.dv.most_similar(positive=pred_vec, topn=1))[0][0])

print(y_pred)

print("Accuracy: %.10f%%" % (accuracy(y_test, y_pred)*100))
#for i in range(0, number_clades+1):
#    print(confusion_matrix(y_test, y_pred)[i])


#  vec_size: 300, ep: 50
