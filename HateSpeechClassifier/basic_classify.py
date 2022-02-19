# Simple Classifier [Linear Regression] 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv # for excel too 
import json 
import random
import numpy as np
import itertools 
import matplotlib as mpl
import sys
import os
import copy 
import pickle 
from scipy import sparse
# import scipy

# for natural language processing 
import nltk
from nltk.stem.porter import *
# nltk.download("stopwords")
# nltk.download("averaged_perceptron_tagger")

from sklearn.preprocessing import StandardScaler 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import mixture, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier # similar to Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import f1_score, confusion_matrix, multilabel_confusion_matrix, \
    classification_report, accuracy_score 
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.covariance import empirical_covariance

# import word embeddings models 
from transformers import BertTokenizer, BertModel
import torch

# LSTM and CNN *** sharfard paper 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import LSTM

from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

class Classifier:
    def __init__(self, labels, *args, **kwargs): # X, y, **kwargs):   
        # second_capital = re.search('^([^A-Z]*[A-Z]){2}', kwargs.)
        self.labels = labels # kwargs.pop('labels', '') # args[1] 
        self.X = kwargs.get('X', None)
        self.y = kwargs.get('y', None)
        name_regex = '([a-zA-Z]*)(\s*)([a-zA-Z]*)'
        clsf = kwargs.pop('classifier', '')
        normalize = lambda name: (name.group(1).strip().capitalize() + " " + name.group(3).strip().capitalize())
        # print(re.match(name_regex, clsf).group(1).strip().capitalize())
        # print(re.match(name_regex, clsf).group(3).strip().capitalize())
        self.model_name = re.sub(name_regex, normalize, clsf).strip() # or re.sub(name_regex, normalize, kwargs.pop('model', ''))
        # print(clsf)
        self.model = LogisticRegression(random_state = 0,  solver = 'lbfgs', warm_start = True) if self.model_name == "Logistic Regression" else \
            RandomForestClassifier(random_state = 0) if self.model_name == "Random Forest" else \
            GaussianNB() if self.model_name == "Naive Bayes" else \
            DecisionTreeClassifier(random_state = 0) if self.model_name == "Decision Trees" else \
            XGBoostClassifier(random_state = 0) if self.model_name == "XGBoost" else \
            SVC(random_state = 0) # if self.model_name == "SVM" else None
        if self.model is None:
            self.model = kwargs.get('model', None) # if self.model_name == "SVM" else None
        self.model_params = kwargs.get('model_params', None) # == 'model_params'
        self.CV = kwargs.get('cv', None) # == 'cv'
        self.cv_params = kwargs.get('cv_params', None) # == 'cv_params'
        # print(self.model_params)
        # print(self.CV)
        # print(self.cv_params)
        self.preds = None
        self.cls_report = None 
        self.acc_report = None 
        self.train_score = None
        self.test_score = None 
        self.conf_m = None 
        # self.features_n = len(tfidf_vectorizer.vocabulary_)
        self.filename = (self.model_name.lower().replace(" ", "_")) 
        # print(param_grid)
        # print("args:", args)
        # print("kwargs:", kwargs)
        # print("cv params:", self.cv_params)
        if self.CV is not None: 
            self.cross_validation(*args, **kwargs) #model = self.model, model_params = self.model_params, 
                              # cv = self.CV, cv_params = self.cv_params)
      
    def cross_validation(self, *args, **kwargs): 
        model = self.model if self.model is not None else kwargs.pop('model', None)
        model_params = self.model_params if self.model_params is not None else kwargs.pop('model_params', None)
        cv = self.CV if self.CV is not None else kwargs.pop('cv', None)
        cv_params = self.cv_params if self.cv_params is not None else kwargs.pop('cv_params', None)
        print("cross_validation")
        # print(model)
        print("args", args)
        print("kwargs:", kwargs)
        print("cv params:", cv_params)
        # print(param_grid)
        args = [cv_params[0]] 
        default_args = cv_params[1] # [1:] 
        print("param_grid:", param_grid)
        print("cv params:", cv_params)
        print("args", args)
        print("default_args", default_args)
        self.model = cv(model, *args, **default_args) # *args, **default_args) # create cross validator model 
        # self.model = cv(model, args, kwargs, *cv_params) # args, kwargs, *cv_params)
        # self.CV = cv(model, args, kwargs) if (model_params and cv_params) is None else cv(model(*model_params), args, kwargs) if cv_params is None else \
        #     cv(model, args, kwargs, *cv_params) if model_params is None else cv(model(*model_params), args, kwargs, *cv_params)  # GridSearch()
        return self.model 
    
    def set_params(self, *args, **kwargs):
        self.labels = kwargs.pop('labels', '')# should be labels 
        
        name_regex = '([a-zA-Z]*)(\s*)([a-zA-Z]*)'
        clsf = kwargs.pop('classifier', '')
        normalize = lambda name: (name.group(1).strip().capitalize() + " " + name.group(3).strip().capitalize())
        self.model_name = re.sub(name_regex, normalize, clsf).strip() # or re.sub(name_regex, normalize, kwargs.pop('model', ''))
        self.model = LogisticRegression(random_state = 0) if self.model_name == "Logistic Regression" else \
            RandomForestClassifier(random_state = 0) if self.model_name == "Random Forest" else \
            GaussianNB() if self.model_name == "Naive Bayes" else \
            DecisionTreeClassifier(random_state = 0) if self.model_name == "Decision Trees" else \
            XGBoostClassifier(random_state = 0) if self.model_name == "XGBoost" else \
            SVC(random_state = 0) 
        if self.model is None:
            self.model = kwargs.pop('model', None) # if self.model_name == "SVM" else None
        self.model_params = kwargs.pop('model_params', None) # == 'model_params'
        self.CV = kwargs.pop('cv', None) # == 'cv'
        self.cv_params = kwargs.pop('cv_params', None) # == 'cv_params'
        
        self.preds = kwargs.pop('preds', None)
        self.score = kwargs.pop('score', None)
        self.cls_report = kwargs.pop('cls_report', None) 
        self.acc_report = kwargs.pop('acc_report', None)
        self.conf_m = None 
        self.filename = (self.model_name.lower().replace(" ", "_")) 
        
    def set_model_params(self, *args, **kwargs): 
        self.model = LogisticRegression(args, kwargs, random_state = 0) if (self.model_name == "Logistic Regression") else \
            RandomForestClassifier(args, kwargs, random_state = 0) if (self.model_name == "Random Forest") else \
            GaussianNB(args, kwargs) if self.model_name == "Naive Bayes" else \
            DecisionTreeClassifier(args, kwargs, random_state = 0) if self.model_name == "Decision Trees" else \
            XGBoostClassifier(args, kwargs, random_state = 0) if self.model_name == "XGBoost" else \
            SVC(args, kwargs, random_state = 0) # if self.model_name == "SVM" else None
        if self.model is None:
            self.model = (kwargs.pop('model', None))(args, kwargs) # if self.model_name == "SVM" else None
  
    def get_model(self): # return current model instance 
        return self.model
    
    def load(self, infile = "", path = ""): 
        if infile is "": 
            with open(path + self.filename + ".pkl", "rb") as file: # read byte 
                self = pickle.load(file)
        else: 
            with open(infile, "rb") as file: # read byte 
                self = pickle.load(file)
        return self # Classifier(labels, model) 
    
    def load_model(self, infile = "", path = ""): 
        if infile is "": 
            with open(path + self.filename + "_model.pkl", "rb") as file: # read byte 
                self.model = pickle.load(file)
        else: 
            with open(infile, "rb") as file: # read byte 
                self.model = pickle.load(file)
        return self.model # Classifier(labels, model) 
    
    def save(self, path = "", outfile = ""):
        if outfile is "": 
            with open(path + self.filename + ".pkl", "wb") as file: # write byte 
                pickle.dump(self, file)
        else: 
            with open(outfile, "wb") as file: # write byte 
                pickle.dump(self, file)
        return self 
    
    def save_model(self, path = "", outfile = ""):
        if outfile is "": 
            with open(path + self.filename + "_model.pkl", "wb") as file: # write byte 
                pickle.dump(self.model, file)
        else: 
            with open(outfile, "wb") as file: # write byte 
                pickle.dump(self.model, file)
        return self.model 
        
    def fit(self, X_train, y_train):  # returns model object as well 
        self.model.fit(X_train, y_train)
        return self.model 
        
    def predict(self, X_test): 
        self.preds = self.model.predict(X_test)
        return self.preds
    
    def predict_proba(self, X_test):
        self.preds_proba = self.model.predict_proba(X_test)
        return self.preds_proba
    
    def score(self, X_test, y_test): 
        self.score = self.model.score(X_test, y_test) 
        return self.score
    
    def report_results(self, y_test = None, preds = None, output = True, save = False, path = "", outfile = ""): 
        cls_report = self.cls_report if (y_test is None) and (preds is None) else self.classification_report(y_test, preds, output = False) 
        acc_report = self.acc_report if (y_test is None) and (preds is None) else self.accuracy_score(y_test, preds, output = False)
        conf_m = np.array_str(self.conf_m, precision = 3) if (y_test is None) and (preds is None) \
        else np.array_str(self.confusion_matrix(y_test, preds, output = False), precision = 3) 
    
        report = cls_report + "\n" + acc_report + "\n" + conf_m

        if save: 
            if outfile is "": 
                # report.to_csv(path + self.filename + "_results.csv", sep = ",", index = False, encoding = 'utf8')
                with open(path + self.filename + "_results.txt", "w+", encoding = "utf8") as file: # write byte 
                    file.write(report)
            else: 
                # report.to_csv(outfile, sep = ",", index = False, encoding = 'utf8')
                with open(outfile, "w+", encoding = "utf8") as file: # write byte 
                    file.write(report)
    
    def classification_report(self, y_test, preds, output = True, save = False, path = "", outfile = ""): 
        self.cls_report = classification_report(y_test, preds)
        if output: 
            print(self.cls_report)
        if save: 
            report = classification_report(y_test, preds, output_dict = save)
            report = pd.DataFrame(report).transpose()
            if outfile is "": 
                report.to_csv(path + self.filename + "_clsf_report.csv", sep = ",", index = False, encoding = 'utf8')
                # with open(path + self.filename + ".csv", encoding = "utf8") as file: # write byte 
                    # file.write(report)
            else: 
                report.to_csv(outfile, sep = ",", index = False, encoding = 'utf8')
                # with open(outfile, encoding = "utf8") as file: # write byte 
                    # file.write(report) 
        return self.cls_report 
            
    def accuracy_score(self, y_test, preds, output = True, save = False, path = "", outfile = ""): 
        self.train_score = self.model.score(X_train, y_train)
        self.test_score = accuracy_score(y_test, preds)
        report = '{} Train accuracy {:.3f}%'.format(self.model_name, self.train_score * 100) + '\n' \
            + '{} Test accuracy {:.3f}%'.format(self.model_name, self.test_score * 100) + '\n'
        self.acc_report = report 
        if output: 
            print(report) 
        if save: 
            if outfile is "": 
                # report.to_csv(path + self.filename + "acc_report.csv", sep = ",", index = False, encoding = 'utf8')
                with open(path + self.filename + "_acc_report.txt", "w+", encoding = "utf8") as file: # write byte 
                    file.write(report)
            else: 
                # report.to_csv(outfile, sep = ",", index = False, encoding = 'utf8')
                with open(outfile, "w+", encoding = "utf8") as file: # write byte 
                    file.write(report) 
        return self.acc_report   
        
    def confusion_matrix(self, y_test, preds, output = True, save = False, path = "", outfile = ""): 
        self.conf_m = confusion_matrix(y_test, preds)
        if output:  
            print('Confusion matrix: ')
            print(self.conf_m)
        
        if save: 
            if outfile is "": 
                # report.to_csv(path + self.filename + "conf_matrix.csv", sep = ",", index = False, encoding = 'utf8')
                with open(path + self.filename + "_conf_matrix.txt", "w+", encoding = "utf8") as file: # write byte 
                    file.write(str(self.conf_m))
            else: 
                # report.to_csv(outfile, sep = ",", index = False, encoding = 'utf8')
                with open(outfile, "w+", encoding = "utf8") as file: # write byte 
                    file.write(str(self.conf_m)) 
        return self.conf_m
    
    def heatmap(self, conf_m = None, labels = None, save = False, path = "", outfile = ""):  # draw confusion matrix 
        conf_m = self.conf_m if conf_m is None else conf_m 
        labels = self.labels if labels is None else labels 
        
        size = len(conf_m)
        matrix = np.zeros((size, size))
        for i in range(0, size):
            matrix[i, :] = (conf_m[i, :])/(float(conf_m[i,:].sum()))
            
        conf_df = pd.DataFrame(matrix, index = labels, columns = labels)
        plt.figure(figsize=(size * 1.5, size * 1.25))
        sns.heatmap(conf_df, annot = True, annot_kws = {"size": size * 1.33},
                        cmap='gist_gray_r', cbar = False, square = True, fmt = '.2f')
        plt.ylabel('True categories', fontsize = size * 1.5)
        plt.xlabel('Predicted categories', fontsize = size * 1.5)
        plt.tick_params(labelsize = size * 1.33)

        if save:
            f_name = self.filename + "_heatmap" # "_".join(self.model_name.lower())
            if outfile is "":
                plt.savefig(path + f_name + ".pdf") # '.pdf')
            else: 
                plt.savefig(outfile)

def _test_model(X, model, labels):
    if type(X) is not list: 
        x_test = [X]
        # X_test = X.split(" ") # make list of words 
    else: 
        x_test = X
    # print(X_test)
    feature_num = 5733 # len(tfidf_vectorizer.vocabulary_) # model.features_n
    # corpus_docs = pd.concat([pd.DataFrame(x_test), corpus]).reset_index(drop = True) #  columns = ["HATE_SPEECH"])], axis = 0)
    corpus_docs = corpus
    
    for i in range(len(x_test)):
        corpus_docs.iloc[i] = x_test[i]
        # test_docs.iloc[i] = x_test[i]
#     display(test_docs)
    # display(corpus_docs)
    tfidf_vectorizer = TfidfVectorizer(ngram_range = (1, 2),
                                       max_df = 0.75, min_df = 3, # 0.75, 5 
                                       max_features = feature_num) # 10000)
    docs = tfidf_vectorizer.fit_transform(corpus_docs) # corpus

    features = tfidf_vectorizer.get_feature_names() # _out(input_features = None)
    # print(len(features))
    # feature_num = len(features)
    # print(docs)
    
    X_train, X_test, train_docs, test_docs = train_test_split(docs, corpus_docs, random_state = 0, test_size = 0.2)
    
    X_test_arr = (X_test.toarray())
    for i in range(len(x_test)):
        # print((X_test_arr)[i,:])
        # print()
        # print(docs.toarray()[i,:])
        X_test_arr[i,:] = docs.toarray()[i,:]
        test_docs.iloc[i] = x_test[i]
        
    X_test = sparse.csr_matrix(X_test_arr)
    # print(X_test)
    y_preds = model.predict(X_test)
    y_preds_acc = model.predict_proba(X_test)
    
    pred_table_cols = ["TEXT"]
    pred_table_cols.extend(label_names)
    pred_table = pd.concat([pd.DataFrame(test_docs.reset_index(drop = True)), pd.DataFrame(y_preds_acc * 100)], axis = 1)
    pred_table.columns = pred_table_cols
    pred_table["prediction"] = y_preds # (pd.DataFrame(y_preds2, "CLASS"))
    print("Test table: ")
    display(pred_table.iloc[:len(x_test)])

    print("Full prediction table: ")
    display(pred_table)
    return (y_preds, y_preds_acc)
    
def test_model(X): 
    return _test_model(X, model, labels) 

def main(): 
    # Defining globals
    global save 
    global choice
    global labels
    global model 
    global corpus 
    global combined_hate_list 
    global feature_num 
    
    conan_path = 'CONAN-master/Multitarget-CONAN/'
    conan_file = 'Multitarget-CONAN.json'
    conan_p = conan_path + conan_file

    davidson_path = 'hate-speech-and-offensive-language-master/data/'
    davidson_f = 'labeled_data.csv' 
    davidson_p = davidson_path + davidson_f

    combined_path = ''
    combined_f = 'combined_dataset.csv'
    combined_p = combined_path + combined_f

    reports_path = 'classification_reports/'
    data_path = 'datasets/'
    model_path = 'models/'

    # list of classifiers to test 
    classifiers = ["Logistic Regression", "Random Forest", "Decision Trees", "XGBoost", "SVM", "Naive Bayes"]

    # create lists of names for loading models 
    class_list = [str.lower(clsf).replace(" ", "_") + ".pkl" for clsf in classifiers]
    class_model_list =  [str.lower(clsf).replace(" ", "_") + "_model.pkl" for clsf in classifiers] 
    # print(class_list)
    # print(class_model_list)
    
    # list of label names 
    label_names = ["DISABLED", "JEWS", "LGBT+", "MIGRANTS", "MUSLIMS", "POC", "WOMEN", "other", "none"]

    with open(data_path + "combined_dataset.csv", "r", encoding = "utf8") as file: 
        combined_tar_df = pd.read_csv(file)
        file.close() 

    combined_hate_list = combined_tar_df.iloc[:, 0]
    corpus = combined_hate_list
#     combined_tar_list = combined_tar_df.iloc[:, 1]
#     # display(combined_tar_list)

#     # get list of 9 labels from saved datasets
#     labels = np.unique(combined_tar_list)
    labels = [i for i in range(len(label_names))]
    
    # Simple classifier that tests all known classifiers we have 
    # make True if you want to see output in the console 
    output = False 

    # make True if you want to save models, reports, and images to specified directories 
    save = False 
    
    # make value that indexes classifiers list 
    choice = 0 
    
    '''Load last updated classifier object''' 
    model = Classifier(labels).load(infile = model_path + class_list[choice])
    # print(model.get_model()) 

    # Not necessary with pre-loaded model (using .load() function 
    # model.fit(X_train, y_train)
    
    '''Option 1 is make your own python file with new data/tweets/words to test. Import the test_model function to do so, but read the notes I left here'''
    '''Option 2 is use my main function and this model called with some string or list of strings X for testing''' 

    # Can be a string or a list of strings
    # Fundamental problem with testing only one string at a time, but I made a hacky solution
    # Also don't test more than 1834 tweets/posts at a time, it will break 
    X = ["Happy birthday bird! @codyflitcraft",
         "Haha, Yankees fans are bent out of shape about the Hu27le graphic",
         "Maybe the UN could talk to those asian and african nations responsible for \
    # 90%+ of the pollution in the oceans' instead of insisting on this bullshit about climate change.# ",
        ]
        # "This is a very long sentence that *@ will likely not look like a tweet.!!! " 
        # ]
    
    # returns prediction column and probability for the 8 classes (numbered 0 - 8 from left to right) 
    y_preds, y_preds_acc = test_model(X)
    
    # Full prediction table is returned because model has to be tested on same number of inputs as it was originally tested on. It's a problem
    
main() 
                