import json, pickle, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from typing import List, Dict, Tuple
from collections import defaultdict

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from Embedding import Embedding, utils_preprocess_text
from Model import Model, MODELS_PATH

SEED = 100
TARGET_TYPES = ['DISABLED', 'JEWS', 'LGBT+', 'MIGRANTS', 'MUSLIMS', 'POC', 'WOMEN', 'other', 'None']
TARGET_TO_INDEX = {'DISABLED': 0,
                   'JEWS': 1,
                   'LGBT+': 2,
                   'MIGRANTS': 3,
                   'MUSLIMS': 4,
                   'POC': 5,
                   'WOMEN': 6,
                   'other': 7,
                   'None': 8}

def run_tests(texts: List[str], labels: List[int], model_names: List[str], debug=True) -> dict:
    '''
    Get metrics for multiple embedding/model combos. 
    Assumes these models are already saved.
    texts and labels are unseen.
    No training is done.

    params:
        texts: list of strings (posts), already stripped of punctuation/symbols
        labels: class labels
        model_names: filenames of saved models to test; assumes these are in MODELS_PATH
    
    returns: dictionary of results:
        {'model_name': {
            'report': classification report,
            'accuracy': test accuracy,
            'confusion' : confusion matrix
            }
        }
    '''
    results = {}
    for model_name in model_names:
        if debug:
            print('----------------------------------------')
            print(model_name)

        # classify given texts with current model
        predicted_labels, predicted_probs = classify(texts, model_name)
        predicted_labels = [TARGET_TO_INDEX[target] for target in predicted_labels]
        
        # get classification report, test accuracy, confusion matrix
        report = classification_report(labels, predicted_labels, output_dict=True)
        accuracy = accuracy_score(labels, predicted_labels)
        balanced_accuracy = balanced_accuracy_score(labels, predicted_labels)
        precision = precision_score(labels, predicted_labels, average='weighted')
        recall = recall_score(labels, predicted_labels, average='weighted') 
        f1 = f1_score(labels, predicted_labels, average='weighted') 
        #roc_auc = roc_auc_score(labels, predicted_labels, multi_class='ovr')
        
        confusion = confusion_matrix(labels, predicted_labels)

        if debug:
            print(f'Accuracy: {accuracy * 100}')
            print(report)
            print(confusion)

        results[model_name] = {
            'report': report,
            'accuracy': accuracy,
            'balanced accuracy': balanced_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            #'roc_auc': roc_auc,
            'confusion' : confusion
        }
    
    return results


def run_tests_kfold(texts: List[str], labels: List[int],
                    embedding_types=['tfidf', 'word2vec', 'doc2vec', 'glove', 'bert'],
                    model_types=['LR', 'RF', 'DT', 'SVM', 'XGB'],
                    with_stopwords=False,
                    weighting_type='equal',  
                    dimensions=300,
                    scoring = ['accuracy', 'balanced_accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovr'],
                    debug=True
                    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    '''
    Based on: https://towardsdatascience.com/quickly-test-multiple-models-a98477476f0
    
    Run kfold cross validation on multiple embedding/model combos. 

    params:
        texts: list of strings (posts), already stripped of punctuation/symbols
        labels: class labels
        embedding_types: which embeddings to use; can pass in less than the default
        model_types: which models to use; can pass in less than default
        with_stopwords: whether to remove stopwords from input
        weighting_type: how to weight word vectors for word2vec and glove. Other option is 'tfidf'
        dimensions: for glove and doc2vec. Other options are 50, 100, 200 for glove, any for doc2vec
        scoring: which metrics to evaluate the model on. Make sure these are suited for multiclassifcation
        debug: if True, print current embedding and model that is training    

    returns:
        pandas dataframe summarizing results, for each model and each k in the kfold
        dictionary {model name: confusion matrix}
    '''   
    # remove stopwords if necessary
    if not with_stopwords:
        stop_words = stopwords.words('english')
        texts = [t for t in texts if not t in stop_words]

    dfs = [] # save results for each scoring metric for each k in the kfold
    confusions = {} # save one confusion matrix for each embedding/model combo
    for embedding_type in embedding_types:
        if debug:
            print('----------------------------------------')
            print(embedding_type)

        embedding = Embedding(embedding_type, 
                              with_stopwords=with_stopwords, 
                              weighting=weighting_type,
                              dimensions=dimensions)
        
        # vectorize train and test set
        X = embedding.vectorize(texts, load_train=True)

        # might need to save tfidf vectorizer and matrix for later use
        if embedding_type == 'tfidf' and weighting_type == 'tfidf':
            embedding.save(train_test_split=True)

       #X_test = embedding.vectorize(X_test_text, unseen=True, load_test=True)
       #
       #if embedding_type == 'tfidf' and weighting_type == 'tfidf':
       #    embedding.save(train_test_split=True, save_test=True)

        for model_type in model_types:
            if debug:
                print('  ' + model_type)
            
            # run kfold
            model = Model(model_type, random_state=SEED)
            
            kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=SEED)
            cv_results = model_selection.cross_validate(model.model, X, labels, cv=kfold, scoring=scoring)
            #clf = model.model.fit(X_train, y_train)
            #y_pred = clf.predict(X_test)

            # add results to dataframe
            model_name = model_type + '_' + embedding.get_filename()

            this_df = pd.DataFrame(cv_results)
            this_df['model'] = model_name
            dfs.append(this_df)
            
            # save confusion
            #confusion = confusion_matrix(y_test, y_pred)
            #confusions[model_name] = confusion

            if debug:
                #print(classification_report(y_test, y_pred, target_names=TARGET_TYPES))  
                print(this_df)

    final_df = pd.concat(dfs, ignore_index=True)
    return final_df #, confusions


def train_and_save(texts: List[str], labels: List[int],
                   embedding_types=['tfidf', 'word2vec', 'doc2vec', 'glove', 'bert'],
                   model_types=['LR', 'RF', 'DT', 'SVM', 'XGB'],
                   with_stopwords=False,
                   weighting_type='equal',  
                   dimensions=300,
                   debug=True):
    '''
    Train multiple combinations of embeddings and models on an entire training dataset,
    and save the embeddings and models.

    params:
        texts: list of strings (posts), already stripped of punctuation/symbols
        labels: class labels
        embedding_types: which embeddings to use; can pass in less than the default
        model_types: which models to use; can pass in less than default
        with_stopwords: whether to remove stopwords from input
        weighting_type: how to weight word vectors for word2vec and glove. Other option is 'tfidf'
        dimensions: for glove and doc2vec. Other options are 50, 100, 200 for glove, any for doc2vec
        debug: if True, print current embedding and model that is training

    returns: none; saves each model to file

    file naming scheme: model_embedding_weighting_epochs_dimensions_stopwords
        note, not every embedding will have every one of these parameters
    '''
    # remove stopwords if necessary
    if not with_stopwords:
        stop_words = stopwords.words('english')
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

            model = Model(model_type, random_state=SEED)
            model.model.fit(X, labels)
            model.save(embedding_filename)


def classify(texts: List[str], model_path: str) -> Tuple[List[str], List[float]]:
    '''
    Classify unseen text

    params:
        texts: list of strings (social media posts)
        model_path: string, path to saved model to use for classification

    returns: parallel lists of predicted classes, predicted probabilites
    '''
    filename = model_path.split('/')[-1]
    model_name, *embedding_params = filename.split('_')
    embedding_filename = '_'.join(param for param in embedding_params)
    embedding_type = embedding_params[0]

    if filename.endswith('nostop'):
        stop_words = stopwords.words('english')
    else:
        stop_words = None
    
    # preprocess input
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

    for i, probs in enumerate(probabilities):
        max_prob = max(probs)
        max_class = np.where(probs == max_prob)[0][0]
        predicted_classes[i] = TARGET_TYPES[max_class]
        predicted_probs[i] = max_prob

    return predicted_classes, predicted_probs

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
    '''
    # train on the above data and save multiple combinations of embeddings/models 
    #train_and_save(tweets, labels)

    df, confusion = run_tests_kfold(tweets, labels, embedding_types=['tfidf'], model_types=['LR'])
    print(df)
    print(confusion)
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
    