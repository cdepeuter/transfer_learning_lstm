# Using transfer learning to predict review sentiment and category

In this paper, we simulate an environment where a user is interested in performing a classification task on a small text dataset, while a larger, similar text dataset is available. Because Deep Learning approaches perform better on large datasets, we compare performance of models built on the smaller dataset itself versus a model built on the larger dataset and then transferred to the smaller one. Specifically, we implement two approaches to classify Yelp reviews as positive- or negative-sentiment: First, an LSTM-based model trained on a dataset of Amazon reviews and evaluated on a small dataset of Yelp reviews, and then an LSTM model trained and evaluated on the Yelp dataset itself. For very small samples of the Yelp dataset, the transfer approach is better; for very large samples, the self-trained model performs better. The first guiding question for our project is: How big must the Yelp dataset be in order for it to be preferred to train on Yelp itself rather than rely on transfer learning? We then evaluate the tenability of implementing a transfer approach across tasks, where a network trained to classify sentiment on Amazon reviews is re-trained to another task for Yelp.

## Data Cleaning (https://github.com/cdepeuter/transfer_learning_lstm/blob/master/clean_data.py)

We first clean the raw reviews for each source. Cleaning involves lemmatization, lower casing all words, and the sentiment score to binary mapping. This process reads from the data/<source>/raw and writes to the data/<source>/cleaned folder

`$ python clean_data.py amazon` 
`$ python clean_data.py yelp` 


## Building a vocabulary (https://github.com/cdepeuter/transfer_learning_lstm/blob/master/make_vocab.py) 

In this project we use two different vocabularies, one standard vocabulary using the most frequent 400,000 words in the google news embedding

### Finding domain-independent sentiment words (https://github.com/cdepeuter/transfer_learning_lstm/blob/master/get_domain_independent_sentiments.py)

We take the dataset (https://nlp.stanford.edu/projects/socialsent/) from the Stanford CoreNLP group which describes the polarity of words for different subreddits, and find the top 2000 words that occur in the must subreddit polarity lists, we consider this as a list of domain-independent polarized words. We build a vocabulary from GoogleNews in the same way as before, but not allowing for any of these words.


## Review embedding (https://github.com/cdepeuter/transfer_learning_lstm/blob/master/get_clean_embeds.py)

The next step is to get embeddings for each review. To do this we use the GoogleNews word2vec embedding. For each review we take the embedding for the first 100 words in each review that is in the defined vocabulary, this is the input for our LSTM.

## Amazon-trained LSTM: (https://github.com/cdepeuter/transfer_learning_lstm/blob/master/lstm_gcp.py)

A standard Tensorflow LSTM implementation with inputs for batch size, LSTM units, and iterations. A we used Google Cloud Platform to grid search across these parameters (https://github.com/cdepeuter/transfer_learning_lstm/blob/master/grid_search_lstm.py).

`$ python lstm_gcp.py 512 96 5000` <- the best model, achieving 96% accuracy
`$ python lstm_gcp.py 1024 48 5000`
`$ python lstm_gcp.py 1024 96 5000`

## Yelp-trained LSTM (https://github.com/cdepeuter/transfer_learning_lstm/blob/master/yelp_lstm.py)

Since we were interested in 3 different outcome spaces, as well as how well the LSTM would do for different sizes of datasets, this file has inputs for the target variable, as well as the size of data to train on.

`$ python yelp_lstm.py sentiment 15000`
`$ python yelp_lstm.py stars 20000`
`$ python yelp_lstm.py cats 20000`

## Retraining the last layer for Yelp data (https://github.com/cdepeuter/transfer_learning_lstm/blob/master/load_retrain_lstm.py)

This file allows for retraining an inputted LSTM for different targets as well as Yelp data sizes. Example usage.
`$ python load_retrain_lstm.py sentiment 24000`
`$ python load_retrain_lstm.py stars 7500`
`$ python load_retrain_lstm.py cats 2000`


All models were trained using Google Cloud Platform.
