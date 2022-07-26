## bachelorproject

Goal of this project was to create a model that gets a certain part of the rna of a coronavirus as input and uses it to predict the virus' clade. <br />
The first prototype of such a model was a simple sequential model (see model_version_1). A problem for this model was however, that the given data was not only highly imbalanced, but also had the property of the sequences (belonging to different clades) being very similar to each other. Because of this, the model was unable to differentiate between them and simply predicted every sequence to belong to the clade that occured the most in training data. <br />
A first try to change this by simply rebalancing the data using a combination of over- and undersampling did nothing to solve this issue.  <br />
The second approach was to use a doc2vec model to map the sequences to vectors and use these vectors instead of simply the encoded original sequences to train the sequential model. This too lead to no improvement, most likely because the similarity of the vectors belonging to different clades was still to high for the model.







###### explanation of the different files:
 - comp_model: the 'main file', contains the final version of the model, only file that needs to be executed<br />
   &emsp; &emsp; &emsp; &emsp; &emsp; composed of a doc2vec model to 'preprocess' the data sequences and a sequential model
               
 - model_doc2vec: contains function to build a doc2vec model that converts the amino acid sequences into vectors 
 - helper_functions: contains some small functions to encode or evaluate stuff 
 - model_version_1: first version of the model, basically the same as the sequential model in the composite model
 - model_random_data: uses the model from model_version_1 with randomly generated data
 
 - training/test_data_mini_sample:  small examplary dataset in the style of the original dataset, split into training and test data
 
 ###### packages used:
 numpy, pandas, gesim, keras,tensorflow, imblearn, sklearn, multiprocessing
