# Information-Extraction: Named Entity Resolver and Coreference Resolution

Training a Named Entity Resolver and a Coreference Resolver for Arabic.

## Description

Built a named entity recogniser similar to Lample et al,. (2016), and will train and evaluate your system on the benchmark CoNLL 2003 English datasets. Also built a coreference system based on the mention-ranking algorithm proposed by Lee et al (2017). The first part of the notebook shows how to apply coreference resolution to English and then to Arabic.

### Named Entity Resolver
The script defines a single class (NERModel) which contains all the elements needed for a simple named entity recognition system. Here we use the IO format and treat the flat ner as the sequence labelling task. Unlike BIO format, the IO format does not differentiate the start of the entity and the remaining tokens hence simpler.<br>
The word_embeddings contains the word embeddings of the tokens in the sentences and
are stored as batches of sentences padded to the same length. Please note that for Keras
Input apart from the shape you specified it will always have one more dimension for the batch,
so the actual shape for word_embeddings is [batch_size, max_sent_length, embedding_size],
in our case we use a batch size of 100 sentences.
<br>
We create 2 layers of Bidirectional GRUs of 50 units each. Since we need to output for all tokens in the sentence, we set the return_sequence =True. Then for the two hidden layers, we apply a fully connected layer as well as a Dropout layer to achieve recurrent dropout. The ner_scores represent the probability of a token belonging to a NER label. Therefore we complete the model with an output layer of 5 units, one for each of the NER labels (O, PER, ORG, LOC, MISC). Also in order to compute the binary cross-entropy loss, we need to give this final layer a softmax activation function.
<br>
First, we fetch the predictions using argmax. Then we compute the following from true labels: position of the sentence in the batch, start of the word, end of the word and NER label of the word. Then we compute the same for the predicted named entities.
For each predicted value, we compare the predicted word with the NER labels. If the prediction is true, we add 1 to the True Positives counter. Else we add 1 to the false negative and false positive counter for the true label and predicted label respectively. We finally sum up the total true positives, false negatives and false positives to calculate F1, precision and recall of the predicted output.

### Coreferencer Resolver in Arabic
We load the JSON file and fetch the sentences as well as the clusters from it. We then preprocess the text, removing any punctuation and converting it to the unicode format. We then get the mention and the cluster information using the functions defined above. Then the mentions are split into start and end indices, storing them in two separate arrays. 
<br>
The sentences are converted to glove embeddings and padded, while also fetching the start and end for the padded sentences. We then generate the anaphors and antecedent pairs along with their labels, further converting them to arrays. We store all the information in a list and return it.
<br>
We build the model based on the defined specification. We create separate inputs for embeddings and the mention pairs. Then we squeeze the embeddings and apply a dropout to it. We create a two layer BiLSTM, squeeze them and add dropouts. We flatten the embedding outputs with the mention pairs into a 400D tensor. Finally we create an FFN, add a sigmoid activation and squeeze them. See the model summary for a clearer reference. Converting each cluster into tuples. Then we create a dictionary for mention_to_gold which stores the mention and the cluster it belongs to. Then we get the predicted cluster and its mention for each predicted mention pair. Finally we update and run the evaluator.


## Getting Started

### Dependencies

* Python
* Tensorflow
* Keras
* numpy
* metrics
* re
* tabulate
* warnings

### Installing

* Download the notebook and run it in colab or jupyter environment
* The notebook has a cell to install the coref files [this takes approx 20 minutes to download and import as one file]

### Executing program

* Run the notebook in your preferred environment


## Help

If you've found a new bug, go ahead and create a new GitHub issue. Be sure to include as much information as possible so I can reproduce the bug.


## Authors

Prof Massimo Poesio 
<br>Professor of Computational Linguistics 
<br>Queen Mary University of London
