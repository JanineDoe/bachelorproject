import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.utils import np_utils
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from tensorflow import math
from helper_functions import encode_list, confusion_matrix, accuracy

# fix random seed for reproducibility
np.random.seed(7)

# parameters
seq_length = 1273
number_proteins = 20
voc_size = number_proteins + 1
number_clades = 17
embedding_dim = 32  # kinda random8

# generate datasets
X_train, y_train = make_classification(n_samples=2000, n_features=seq_length, n_informative=seq_length, n_redundant=0,
                                       n_repeated=0, n_classes=number_clades+1, n_clusters_per_class=2, flip_y=0,
                                       class_sep=1.0, shuffle=True, random_state=1)

X_test, y_test = make_classification(n_samples=500, n_features=seq_length, n_informative=seq_length, n_redundant=0,
                                     n_repeated=0, n_classes=number_clades+1, n_clusters_per_class=2, flip_y=0,
                                     class_sep=1.0, shuffle=True, random_state=1)


scaler = MinMaxScaler(feature_range=(1, 20))
X_train = (scaler.fit_transform(X_train)).astype(int)
X_test = (scaler.fit_transform(X_test)).astype(int)

y_train_oh = np_utils.to_categorical(y_train, num_classes=number_clades + 1)  # for categorical crossentropy
y_test_oh = np_utils.to_categorical(y_test, num_classes=number_clades + 1)    # for categorical crossentropy

print((X_train[0]))
print((X_train[0]).shape)
print(y_train_oh[0])
print((y_train_oh[0]).shape)



# create the model
model = Sequential()
model.add(Embedding(input_dim=voc_size, output_dim=embedding_dim, input_length=seq_length))
model.add(LSTM(100))
model.add(Dense(number_clades + 1, activation='sigmoid'))  # use softmax?
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train_oh, validation_data=(X_test, y_test_oh), epochs=5, batch_size=16)

# Final evaluation of the model
# scores = model.evaluate(X_test, y_test_oh, verbose=0)
# print("Accuracy: %.10f%%" % (scores[1] * 100))

y_pred = model.predict(X_test)
y_pred_dec = math.argmax(y_pred, axis=1)

print("Accuracy: %.10f%%" % (accuracy(y_test, y_pred_dec) * 100))
for i in range(0, number_clades + 1):
    print(confusion_matrix(y_test, y_pred_dec)[i])