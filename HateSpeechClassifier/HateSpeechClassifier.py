import json, random, pickle, re, os

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

from Embedding import Embedding, utils_preprocess_text
from Model import Model, MODELS_PATH


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




def classify(texts: List[str], model_path: str):
    '''
    Classify unseen text

    params:
        texts: list of strings (social media posts)
        model_path: string, path to saved model to use for classification

    returns: parallel lists of predicted classes, predicted probabilites
    '''
    # preprocess input
    filename = model_path.split('/')[-1]
    model_name, *embedding_params = filename.split('_')
    embedding_filename = '_'.join(param for param in embedding_params)
    embedding_type = embedding_params[0]

    if filename.endswith('nostop'):
        stop_words = stopwords.words('english')
    else:
        stop_words = None
    
    texts = [utils_preprocess_text(text, lst_stopwords=stop_words) for text in texts]

    # transform input text into embedding
    embedding = Embedding(embedding_type, load_filename=embedding_filename)
    vectorized_texts = embedding.vectorize(texts, unseen=True)

    # predict
    full_model_name = model_name + '_' + embedding_filename
    model = Model(model_name, load_filename=full_model_name, max_iter=2000) 
    probabilities = model.model.predict_proba(vectorized_texts)
    
    # return classes with highest probs for each text
    predicted_classes = [0] * len(texts)
    predicted_probs = [0] * len(texts)

    target_types = ['Disabled', 'Jews', 'LGBT+', 'Migrants', 'Muslims', 'POC', 'Women', 'Other/Mixed', 'None']

    for i, probs in enumerate(probabilities):
        max_prob = max(probs)
        max_class = np.where(probs == max_prob)[0][0]
        predicted_classes[i] = target_types[max_class]
        predicted_probs[i] = max_prob

    return predicted_classes, predicted_probs

if __name__ == '__main__':
    '''
    combined_data_path = 'Data/combined_data.csv'
    
    with open(combined_data_path, 'r', encoding='utf8') as f:
        combined = pd.read_csv(f).to_dict()
    
    tweets = [] # these arrays are aligned by index; tweet[i] has label[i]
    labels = []
    for tweet, label in zip(combined['HATE_SPEECH'].values(), combined['CLASS'].values()):

        tweet = utils_preprocess_text(tweet) #, lst_stopwords=stop_words)

        tweets.append(tweet)
        labels.append(label)
    
    # train on the above data and save multiple combinations of embeddings/models 
    train_and_save(tweets, labels)
    '''

    
    texts = ["Happy birthday bird! @codyflitcraft",
        "Haha, Yankees fans are bent out of shape about the Hu27le graphic",
        "Maybe the UN could talk to those asian and african nations responsible for 90%+ of the pollution in the oceans' instead of insisting on this bullshit about climate change. ",
        "Jews are a separate ethnicity, a danger to national security, and should be banned.",
        "Gay men are a danger to children.",
        "We are being invaded, We must fight until we will vanquish immigrants."]
    
    # classify the above texts using every model currently saved in the models directory
    for model_filename in os.listdir(MODELS_PATH):
        model_name = model_filename.split('/')[-1]
        print('----------------------------------------')
        print(model_name + ':\n')
        classes, probs = classify(texts, model_name)
    
        for i, text in enumerate(texts):
            print(text)
            print(f'  Target = {classes[i]}, probability = {probs[i]}')