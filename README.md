# bachelorproject

~ a (failed) attempt to create a model that gets a certain part of the rna of a coronavirus as input and uses it to predict the virus' clade ~


##explanation of the different files:
 - comp_model: the 'main file', contains the final version of the model, only file that needs to be executed
               composed of a doc2vec model to 'preprocess' the data sequences and a sequential model
               
 - model_doc2vec: contains function to build a doc2vec model that converts the amino acid sequences into vectors 
 - helper_functions: contains some small functions to encode or evaluate stuff 
 - model_version_1: first version of the model, basically the same as the sequential model in the composite model
 - model_random_data: uses the model from model_version_1 with randomly generated data
 
 - training/test_data_mini_sample:  small examplary dataset in the style of the original dataset, split into training and test data
