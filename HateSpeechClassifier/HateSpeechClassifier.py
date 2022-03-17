import json 
import random
import pickle
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_roc_curve

from typing import List
from collections import defaultdict

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from Embedding import Embedding, utils_preprocess_text
from Model import Model


'''
possible embeddings:
    embeddings=[tfidf, word2vec, doc2vec, glove, bert] 
    with_stopwords (bool)
    weighting_type=equal (or tfidf)
    dimensions=300 (or 50, 100)
    
possible models:
    models=[LR, RF, CNN, LSTM]

run tests, generate reports:

on training data (k-fold):
in: X (text), y (labels)
    split into train/test
save to files: confusion, roc curve, reports, summary results


on real-world
in: X (text), y (labels)
    no splitting into train/test; compare y pred with given y
save to files: confusion, roc curve, reports, summary results


train and save:
    no train/test split, train on whole dataset, save to file

classify:
for use by the bots
in: [text], path
out: [labels], [probs]
'''

def run_tests(X: List[str], y: List[int], 
              use_k_fold=True,
              save_confusion=True,
              save_ROC=True, 
              save_reports=True,
              save_summary=True):
    pass
    '''
    Generate test results for multiple combinations of embeddings and models.

    params:
    X: list of strings (posts)
    y: class labels
    use_k_fold: set to True when testing performance on training data;
        set to False when testing performance on real-world Tumblr/Reddit posts;
        False assumes there are already-trained models which it can load

    '''



def train_and_save(texts: List[str], labels: List[int],
                   embedding_types=['tfidf', 'word2vec', 'doc2vec', 'glove', 'bert'],
                   with_stopwords=False,
                   weighting_type='equal',  
                   dimensions=300,
                   model_types=['LR', 'RF'],
                   debug=True):
    '''
    Train multiple combinations of embeddings and models on an entire training dataset,
    and save the embeddings and models.

    params:
        texts: list of strings (posts), already stripped of punctuation/symbols
        labels: class labels
        embedding_types: which embeddings to use; can pass in less than the default
        with_stopwords: whether to remove stopwords from input
        weighting_type: how to weight word vectors for word2vec and glove. Other option is 'tfidf'
        dimensions: for glove and doc2vec. Other options are 50, 100, 200 for glove, any for doc2vec
        model_types: which models to use; can pass in less than default
        debug: if True, print current embedding and model that is training

    returns: none; saves each model to file

    file naming scheme: model_embedding_weighting_epochs_dimensions_stopwords
        note, not every embedding will have every one of these parameters
    '''
    # remove stopwords if necessary
    if not with_stopwords:
        texts = [t for t in texts if not t in stop_words]
    
    for embedding_type in embedding_types:
        if debug:
            print('----------------------------------------')
            print(embedding_type)

        # get embeddings for input text
        embedding = Embedding(embedding_type, 
                              with_stopwords=with_stopwords, 
                              weighting=weighting_type,
                              dimensions=dimensions)
        
        embedding_filename = embedding.get_filename()
        X = embedding.vectorize(texts)
        
        # save embeddings
        embedding.save()

        for model_type in model_types:
            if debug:
                print('  ' + model_type)

            model = Model(model_type, max_iter=2000)
            model.model.fit(X, labels)
            model.save(embedding_filename)




def classify(texts: List[str]):
    # preprocess input

    # transform input

    # predict 

    pass

if __name__ == '__main__':
    combined_data_path = 'Data/combined_data.csv'
    
    with open(combined_data_path, 'r', encoding='utf8') as f:
        combined = pd.read_csv(f).to_dict()
    
    tweets = [] # these arrays are aligned by index; tweet[i] has label[i]
    labels = []
    for tweet, label in zip(combined['HATE_SPEECH'].values(), combined['CLASS'].values()):

        tweet = utils_preprocess_text(tweet) #, lst_stopwords=stop_words)

        tweets.append(tweet)
        labels.append(label)
    
    
    train_and_save(tweets, labels)