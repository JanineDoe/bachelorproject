import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from helper_functions import confusion_matrix, accuracy, encode_list, decode_list

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

'''
# trying to balance dataset
# distribution training data:
# [1: 1328, 2: 1005, 3: 63252, 4: 12641, 5: 4623, 6: 912, 7: 22076, 8: 88, 9: 7460, 10: 22, 11: 35, 12: 2021,
# 13: 198, 14: 4691, 15: 2, 16: 4552, 17: 1]

over_sampling_dict = {6: 1000, 8: 500, 10: 500, 11: 500, 13: 500, 15: 500, 17: 500}
under_sampling_dict = {3: 8000, 4: 4000, 5: 1200,  7: 8000, 9: 2000, 12: 700, 14: 1300, 16: 1300}

over_sampler = RandomOverSampler(sampling_strategy=over_sampling_dict)
under_sampler = RandomUnderSampler(sampling_strategy=under_sampling_dict)

X_train_resampled, y_train_resampled = over_sampler.fit_resample(X_train, y_train)
X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_train_resampled, y_train_resampled)
X_train_resampled = decode_list(X_train_resampled)
'''




test = []
for i in range(0, len(y_test)):
    test.append(TaggedDocument(X_test[i], [y_test[i]]))

train = []
for i in range(0, len(y_train)):
    train.append(TaggedDocument(X_train[i], [y_train[i]]))








# create the model
cores = multiprocessing.cpu_count()
model = Doc2Vec(train, vector_size=300, workers=cores, epochs=10, seed=7)
#model.build_vocab([x for x in train])
#model.train(train, total_examples=len(y_train), epochs=10)




#model.save('./model_doc2vec.d2v')

# evaluate model

y_pred = []
for i in range(0, len(y_test)):
    pred_vec = model.infer_vector([X_test[i]])
    y_pred.append((model.dv.most_similar(positive=pred_vec, topn=1))[0][0])

print(y_pred)

print("Accuracy: %.10f%%" % (accuracy(y_test, y_pred)*100))
for i in range(0, number_clades+1):
    print(confusion_matrix(y_test, y_pred)[i])

