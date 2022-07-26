import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.utils import np_utils
from tensorflow import math
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from gensim.models.doc2vec import Doc2Vec
from sklearn.preprocessing import MinMaxScaler
from model_doc2vec import create_d2v_model
from helper_functions import confusion_matrix, accuracy, kmer_encoder

# parameters
seq_length = 1273
number_proteins = 20
voc_size = number_proteins + 1
number_clades = 17
embedding_dim = 32
kmer_size = 20

# create or (load an already created doc2vec model)
d2v_model = create_d2v_model(vector_size=200, number_epochs=20, kmer_size=kmer_size)
#d2v_model = Doc2Vec.load('model_doc2vec.d2v')

# load training and test data
training_data = pd.read_csv('training_data_mini_sample.csv')
test_data = pd.read_csv('test_data_mini_sample.csv')

y_test = test_data['Clade']
y_train = training_data['Clade']

# use d2v model to convert amino acid sequences into vectors
X_train = []
for i in range(0, len(y_train)):
    X_train.append(np.asarray(d2v_model.infer_vector(kmer_encoder(training_data['Sequence'][i], kmer_size))))
    print(i + 1, ' of ', len(y_train))

X_test = []
for i in range(0, len(y_test)):
    X_test.append(np.asarray(d2v_model.infer_vector(kmer_encoder(test_data['Sequence'][i], kmer_size))))
    print(i + 1, ' of ', len(y_test))

# uncomment and adapt values in dictionary to balance the training data
'''
# original distribution training data:
# [1: 1328, 2: 1005, 3: 63252, 4: 12641, 5: 4623, 6: 912, 7: 22076, 8: 88, 9: 7460, 10: 22, 11: 35, 12: 2021,
# 13: 198, 14: 4691, 15: 2, 16: 4552, 17: 1]

over_sampling_dict = {6: 1000, 8: 500, 10: 500, 11: 500, 13: 500, 15: 500, 17: 500}
under_sampling_dict = {3: 2000, 4: 2000, 5: 1000,  7: 2000, 9: 1000, 12: 700, 14: 1000, 16: 1000}

over_sampler = RandomOverSampler(sampling_strategy=over_sampling_dict)
under_sampler = RandomUnderSampler(sampling_strategy=under_sampling_dict)
X_train, y_train = under_sampler.fit_resample(X_train, np.asarray(y_train))
X_train, y_train = over_sampler.fit_resample(X_train, np.asarray(y_train))
'''


# one hot encode labels as needed by categorical crossentropy:
y_train_oh = np_utils.to_categorical(y_train, num_classes=number_clades + 1)
y_test_oh = np_utils.to_categorical(y_test, num_classes=number_clades + 1)


# normalize input for embedding layer
scaler = MinMaxScaler()
print('scaling training data...')
X_train = scaler.fit_transform(X_train)
print('scaling test data...')
X_test = scaler.transform(X_test)

vec_dim = (X_train[0]).shape[0]  # get vector dimension


# build model
model = Sequential()
model.add(Embedding(input_dim=vec_dim * len(y_train), output_dim=embedding_dim, input_length=vec_dim))
model.add(LSTM(100))
model.add(Dense(number_clades + 1, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(np.array(X_train), np.array(y_train_oh), validation_data=(np.array(X_test), np.array(y_test_oh)), epochs=3,
          batch_size=32)

# Final evaluation of the model
scores = model.evaluate(np.array(X_test), np.array(y_test_oh), verbose=0)
print("Accuracy: %.10f%%" % (scores[1] * 100))

y_pred = model.predict(np.array(X_test))
y_pred_dec = math.argmax(np.array(y_pred), axis=-1)

print("Accuracy: %.10f%%" % (accuracy(y_test, y_pred_dec) * 100))
for i in range(0, number_clades + 1):   # 0th row/column of matrix is meaningless (class 0 doesn't exist)
    print(confusion_matrix(y_test, y_pred_dec)[i])
