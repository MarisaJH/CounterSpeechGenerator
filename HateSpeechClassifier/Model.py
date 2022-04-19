import json, pickle, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt # from https://github.com/reiinakano/scikit-plot
import seaborn as sns

import re 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier # similar to Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import mixture, metrics
# from sklearn.datasets import make_classification
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, multilabel_confusion_matrix, \
    classification_report, accuracy_score, plot_roc_curve, RocCurveDisplay

from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.covariance import empirical_covariance

# LSTM and CNN sharfard paper 
import torch
from torch import nn, optim 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Use to reload changed modules
# %load_ext autoreload
# %autoreload 2
from typing import List, Dict, Tuple
from collections import defaultdict

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from Embedding import Embedding, utils_preprocess_text

REPORTS_PATH = 'Reports/'
SEED = 42 # Mo - maybe we should  try and use the same seed number throughout the code? - was 100
MODELS_PATH = 'Models/MultiClassifier/'
models = ["LR", "RF", "DT", "XGB", "SVC", "NB", "LSTM"]
classifiers = ["Logistic Regression", "Random Forest", "Decision Trees", "XGBoost", "SVM", "Naive Bayes", "LSTM"]
TARGET_TYPES = ['Disabled', 'Jews', 'LGBT+', 'Migrants', 'Muslims', 'POC', 'Women', 'Other/Mixed', 'None']

# kwargs are named parameters (makes a dictionary of names to parameters)
class Model: 
    def __init__(self, model_name: str, load_filename = '', debug = False, **kwargs):
        self.labels = kwargs.pop('labels', TARGET_TYPES) # enter a list of the class labels you want to format output
        self.X = kwargs.pop('X', None) # input independent variable data
        self.y = kwargs.pop('y', None) # corresponding target class variables with X

        # formats all variations of name for classifer as 'UC' for input like 'Upper Case' - one line function to format name 
        name_regex = '([a-zA-Z]*)(\s*)([a-zA-Z]*)'
        format_name = lambda name: (name.group(1).strip().capitalize()[0] + name.group(3).strip().capitalize()[0]) if len(name.group(3)) > 2 else (name.group(1).strip().upper()[:3])
        self.model_name = re.sub(name_regex, format_name, model_name).strip() 
        
        # get model parameters (hyper parameters) as a named parameter for each model 
        self.model_params = kwargs.get('model_params', dict())
        
        if debug and self.model_params is not None: 
            print("model", self.model_name)
            print("model parameters", self.model_params)
        # selects normal model given model_name 
        if load_filename != '':
            self.load_model(load_filename)
        else:
            # Using if statements because not all models will share the same parameters when the dictionary is initialized
            self.model = LogisticRegression(solver = 'lbfgs', warm_start = True, **(self.model_params)) if self.model_name == 'LR' else RandomForestClassifier(**(self.model_params)) \
                       if self.model_name == 'RF' else GaussianNB(**(self.model_params)) \
                       if self.model_name == 'NB' else DecisionTreeClassifier(**(self.model_params)) if self.model_name == 'DT' else XGBClassifier(**(self.model_params)) \
                       if self.model_name == 'XGB' else SVC(probability = True, **(self.model_params)) #\
                       # if self.model_name == 'SVC' else LSTM(self.X, self.y, **(self.model_params)) # Support Vector Classifier 

        # get Cross Validator object as a named parameter and parameters for Cross Validator
        self.CV = kwargs.get('cv', None) 
        self.cv_params = kwargs.get('cv_params', None) 
        
        if debug and self.CV is not None: 
            print("cross validator", self.CV)
            print("cross validator parameters", self.cv_params)
        # Retain model evaluation metrics
        self.preds = None
        self.cls_report = None 
        self.acc_report = None 
        self.train_score = None
        self.test_score = None 
        self.conf_m = None  # confusion matrix for test data 
        self.filename = kwargs.get('filename', (self.model_name.lower().replace(" ", "_")))  # format filenames as 'LR', 'XGB', etc. 
        
        # Keep to compare metrics if necessary 
        self.untuned_model = self.model 
        if self.CV is not None and self.cv_params is not None: 
            self.CV, self.untuned_model, self.model = self.cross_validation(debug = debug, **(self.model_params), **(self.cv_params)) 
            
    # For recovering tuned and finalized models 
    def load_model(self, infile = "", path = ""): 
        if infile is "": 
            with open(path + self.filename + "_model.pkl", "rb") as file: # read byte 
                self.model = pickle.load(file)
        else: 
            with open(infile, "rb") as file: # read byte 
                self.model = pickle.load(file)
        return self.model 
    
    # For saving tuned and finalized models 
    def save_model(self, path = "", outfile = ""):
        if outfile is "": 
            with open(path + self.filename + "_model.pkl", "wb") as file: # write byte 
                pickle.dump(self.model, file)
        else: 
            with open(outfile, "wb") as file: # write byte 
                pickle.dump(self.model, file)
        return self.model 
    
    # For recovering params and variables of Model object - special case rarely use 
    def load(self, infile = "", path = ""):  # load Classifier object and model from pkl
        if infile is "": 
            with open(path + self.filename + ".pkl", "rb") as file: # read byte 
                self = pickle.load(file)
        else: 
            with open(infile, "rb") as file: # read byte 
                self = pickle.load(file)
        return self 
        
    # For saving params and variables of Model object - special case - rarely use
    def save(self, path = "", outfile = ""):
        if outfile is "": 
            with open(path + self.filename + ".pkl", "wb") as file: # write byte 
                pickle.dump(self, file)
        else: 
            with open(outfile, "wb") as file: # write byte 
                pickle.dump(self, file)
        return self 

    # Defines cross_validation function to test models: mainly k-fold and called by class but can also be called outside 
    def cross_validation(self, debug = False, **kwargs): 
        model = self.model if self.model is not None else kwargs.pop('model', None)
        cv = self.CV if self.CV is not None else kwargs.pop('cv', None)
        cv_params = self.cv_params if self.cv_params is not None else kwargs.pop('cv_params', None)
        output = kwargs.pop('output', False)
        save = kwargs.pop('save', False) 
        path = kwargs.pop('path', "") 
        if debug: 
            print("cross_validation")
            print(" model:", self.model_name)
            print(" ", model)
            print(" kwargs:", kwargs)
            print(" cv params:", cv_params)
        
        self.CV = cv(model, **cv_params) # *args, **default_args) # create cross validator model
        self.CV.fit(self.X, self.y) # will do StratifiedK-Fold for us rather than K-fold 

        if debug: 
            print("After cross_validation") 
            print(" model: ", self.model_name)
            print(" ", model)
            print(" cross validator: ", self.CV) 
        
        self.model = self.CV 
        self.model = self.CV.best_estimator_  # replace old model with new best model 
        self.model_score = self.CV.best_score_ 
        self.model_params = self.CV.best_params_
        self.cv_report = pd.DataFrame(self.CV.cv_results_)
    
        return self.CV, model, self.model # return cross validator, untuned model, and tuned model 
    
    def get_params(self, debug = False): # return current best model, score, and parameters compared to other models  
        return self.model, self.model_score, self.model_params   
    # Function still needs testing - only use this function if you want to test and tune a model you recently trained and haven't saved yet
    def set_params(self, model_name: str, debug = False, **kwargs):
        name_regex = '([a-zA-Z]*)(\s*)([a-zA-Z]*)'
        format_name = lambda name: (name.group(1).strip().capitalize()[0] + name.group(3).strip().capitalize()[0]) if len(name.group(3)) > 2 else (name.group(1).strip().capitalize()[:3])
        self.model_name = re.sub(name_regex, format_name, model_name).strip() 
        self.labels = kwargs.pop('labels', '') 
        
        if load_filename != '':
            self.load_model(load_filename)
        else:
            self.model = LogisticRegression(solver = 'lbfgs', warm_start = True, **(self.model_params)) if self.model_name == 'LR' else RandomForestClassifier(**(self.model_params)) if self.model_name == 'RF' else GaussianNB(**(self.model_params)) \
                       if self.model_name == 'NB' else DecisionTreeClassifier(**(self.model_params)) if self.model_name == 'DT' else XGBClassifier(**(self.model_params)) \
                       if self.model_name == 'XGB' else SVC(probability = True, **(self.model_params))

        self.CV = kwargs.pop('cv', None)
        self.cv_params = kwargs.pop('cv_params', None) 
        
        self.preds = kwargs.pop('preds', None)
        self.score = kwargs.pop('score', None)
        self.cls_report = kwargs.pop('cls_report', None) 
        self.acc_report = kwargs.pop('acc_report', None)
        self.conf_m = None 
        self.filename = (self.model_name.lower().replace(" ", "_")) 
        
    def get_model(self): # return current model instance 
        return self.model
    
    def fit(self, X_train, y_train): # returns model object and fits model on training input 
        self.model.fit(X_train, y_train)
        return self.model 
        
    def predict(self, X_test): 
        self.preds = self.model.predict(X_test) # returns target predictions for testing input 
        return self.preds
    
    def predict_proba(self, X_test):
        self.preds_proba = self.model.predict_proba(X_test) # returns probabilities for target predictions  
        return self.preds_proba
    
    def score(self, X_test, y_test): 
        self.score = self.model.score(X_test, y_test) # scores model 
        return self.score
    
    # only call after all other report functions 
    def report_results(self, y_test = None, preds = None, output = True, save = False, path = "", outfile = ""):  
        cls_report = self.cls_report if (y_test is None) and (preds is None) else self.classification_report(y_test, preds, output = output) 
        acc_report = self.acc_report if (y_test is None) and (preds is None) else self.accuracy_score(y_test, preds, output = output)
        conf_m = np.array_str(self.conf_m, precision = 3) if (y_test is None) and (preds is None) \
        else np.array_str(self.confusion_matrix(y_test, preds, output = output), precision = 3) 
    
        report = cls_report + "\n" + acc_report + "\n" + conf_m
        if output: 
            print(report) 
        if save: 
            if outfile is "": 
                # report.to_csv(path + self.filename + "_results.csv", sep = ",", index = False, encoding = 'utf8')
                with open(path + self.filename + "_results.txt", "w+", encoding = "utf8") as file: # write byte 
                    file.write(report)
            else: 
                # report.to_csv(outfile, sep = ",", index = False, encoding = 'utf8')
                with open(outfile, "w+", encoding = "utf8") as file: # write byte 
                    file.write(report)
                    
    # only call after cross_validation is called 
    def report_cv_results(self, output = True, save = True, path = "", outfile = ""):  
        self.cv_report = pd.DataFrame(self.CV.cv_results_)
        if output: 
            print(self.cv_report)
        if save: 
            report = self.cv_report
            if outfile is "": 
                report.to_csv(path + self.filename + "_cv_report.csv", sep = ",", index = False, encoding = 'utf8')
            else: 
                report.to_csv(outfile, sep = ",", index = False, encoding = 'utf8')

        return self.cv_report 
    
    # returns classification report of f1, precision, and accuracy metrics to score models 
    def classification_report(self, y_test, preds, output = True, save = False, path = "", outfile = ""): 
        self.cls_report = classification_report(y_test, preds)
        if output: 
            print(self.cls_report)
        if save: 
            report = classification_report(y_test, preds, output_dict = save)
            report = pd.DataFrame(report).transpose()
            if outfile is "": 
                report.to_csv(path + self.filename + "_clsf_report.csv", sep = ",", index = False, encoding = 'utf8')
            else: 
                report.to_csv(outfile, sep = ",", index = False, encoding = 'utf8')
        return self.cls_report 
    
    # returns train and test accuracy of the model
    def accuracy_score(self, y_test, preds, output = True, save = False, path = "", outfile = ""): 
        self.train_score = self.model.score(self.X, self.y) # X and y are the training/target data 
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
        
    ''' returns confusion matrix report of number of positive predictions out of total data points for each class '''
    def confusion_matrix(self, y_test, preds, output = False, save = False, path = "", outfile = ""): 
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
    
    def heatmap(self, conf_m = None, labels = None, num = False, save = False, path = "", outfile = "", model_num = ""):  # draw confusion matrix
        '''
        Draws confusion matrix as a labeled percentage heatmap of positive predictions to total data points 

        params:
            conf_m: can choose to provide a confusion matrix returned by confusion_matrix function, but must run that function first 
            labels: can choose what the target labels for our predictions will be called
            num: If true, also produces a heatmap labeled with total number of positive predictions 
            save: If true, saves to a generated file from the model file name or a given specified 'outfile' 
        '''
        conf_m = self.conf_m if conf_m is None else conf_m 
        labels = self.labels if labels is None else labels 
        
        size = len(conf_m)
        matrix = np.zeros((size, size))
        for i in range(0, size):
            matrix[i, :] = (conf_m[i, :])/(float(conf_m[i,:].sum())) # calculate percentage matrix of true positives
            
        conf_df = pd.DataFrame(matrix, index = labels, columns = labels)
        
        plt.figure(figsize=(size * 1.5, size * 1.25))
        sns.heatmap(conf_df, annot = True, annot_kws = {"size": size * 1.33},
                        cmap = "YlGnBu", # 'gist_gray_r', 
                    cbar = False, square = True, fmt = '.2f')
        
        plt.ylabel('True categories', fontsize = size * 1.5)
        plt.xlabel('Predicted categories', fontsize = size * 1.5)
        plt.tick_params(labelsize = size * 1.33)
        
        if save:
            f_name = self.filename + "_heatmap" # "_".join(self.model_name.lower())
            
            if outfile is "":
                plt.savefig(path + f_name + ".pdf") # '.pdf')
            else: 
                plt.savefig(outfile)
        
        # second heatmap with category numbers rather than percentile 
        if num: 
            conf_df = pd.DataFrame(conf_m, index = labels, columns = labels)
            plt.figure(figsize = (size * 1.5, size * 1.25))
            sns.heatmap(conf_df, annot = True, annot_kws = {"size": size * 1.33},
                        cmap = "YlGnBu", # 'gist_gray_r', 
                    cbar = False, square = True, fmt = '5d')
            
            plt.ylabel('True categories', fontsize = size * 1.5)
            plt.xlabel('Predicted categories', fontsize = size * 1.5)
            plt.tick_params(labelsize = size * 1.33)
            
        if save:
            f_name = self.filename + "_heatmap_num" # "_".join(self.model_name.lower())

            if outfile is "":
                plt.savefig(path + f_name + str(model_num) +  ".pdf") # + "_num.pdf") # '.pdf')
            else:
                # print(outfile[:-5] + str(model_num) + ".pdf")
                plt.savefig(outfile[:-5] + str(model_num) + ".pdf")
 
    def roc_curve(self, X_test, y_test, save = False, path = "", outfile = ""):
        '''
            Not entirely working yet (single to multi class is more difficult and have to choose either One Against All or Macroaveraging), so YMLV. Try it out and if it graphs, 90% chance the graph is correct but the really sharp ones are a little off 
        params:
            X_test: can choose to provide the X values used to test the model; used for ROC metric evaluation
            y_test: can choose to provide the corresponding y classes to test the model; used for ROC metric evaluation 
            save: If true, saves to a generated file from the 'path' var + model file name or a given specified 'outfile' 
            path: see 'save', don't use with outfile 
            outfile: see 'save', don't use with path
        '''
        # 
        size = len(self.labels) # compare to number of classes to format and size output 
        y_probs = self.model.predict_proba(X_test)
        skplt.metrics.plot_roc(y_test, y_probs, figsize = (size * 1.5, size * 1.25), title_fontsize = size * 1.75, 
                               text_fontsize = size * 1.5)
        # plt.show()
        # plt.ylabel('True positive rate', fontsize = size * 1.5)
        # plt.xlabel('False positive rate', fontsize = size * 1.5)
        # plt.tick_params(labelsize = size * 1.33)
        
        if save:
            f_name = self.filename + "_roc_curves" # "_".join(self.model_name.lower())
            if outfile is "":
                plt.savefig(path + f_name + ".pdf") # '.pdf')
            else: 
                plt.savefig(outfile)
    
# taken from tutorial: https://closeheat.com/blog/pytorch-lstm-text-generation-tutorial
class NeuralNet(nn.Module):
    def __init__(self, model_name = 'LSTM', load_filename = '', X = None, y = None, debug = False, **kwargs):
        # super(NeuralNet(), self).__init__()
        self.debug = debug
        if self.debug: 
            print("NeuralNet: ", model_name)
        self.X = X
        self.y = y 
        self.labels = kwargs.pop('labels', TARGET_TYPES) # enter a list of the class labels you want to format output  
        self.model_name = model_name
        # embedding for Neural Nets
        self.embed = kwargs.pop('embed', Embedding('tfidf'))
        try: # if hasattr(self.embed, 'dimensions'): 
            self.dimensions = self.embed.dimensions
        except AttributeError: 
            self.dimensions = kwargs.pop('dimensions', 300) 
            
        n_vocab = len(X) # len(self.labels)
        self.embedding_dim = 300 # len(y) # 128
        if self.debug: 
            print("Embedding dimensions: ", embed.dimensions)
        self.embedding = kwargs.pop('embed', nn.Embedding(
            num_embeddings = n_vocab,
            embedding_dim = self.embedding_dim,
        )) 
        if embed is not None: 
            self.embedding_dim = 300 # len((y)) # 300 # 128
            
        self.lstm_size = n_vocab # 128
        self.hidden_size = kwargs.pop('hidden_size', self.lstm_size) 
        self.num_layers = kwargs.pop('num_layers', 3) 
        self.dropout = kwargs.pop('dropout', 0.2)
        
        self.lstm = nn.LSTM(
            input_size = self.lstm_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            dropout = self.dropout,
        )
        if self.debug: 
            print("LSTM: ", self.lstm)
            
        self.fc = nn.Linear(self.lstm_size, n_vocab)# len(self.labels))
        
    ''' *** USE THIS TO TRAIN MODEL *** '''
    def forward(self, X, prev_state): # prev_state = (h_n, c_n) - prev hidden and cell states
        embed = self.embedding(X)
        output, prev_state = self.lstm(embed, prev_state) # tuples of hidden and cell states of each element in the sequence 
        logits = self.fc(output)
        return logits, prev_state 
    
    def init_state(self, embedding_dim: int): # embedding_dim = sequence_length 
        return (torch.zeros(self.num_layers, embedding_dim, self.lstm_size),
                torch.zeros(self.num_layers, embedding_dim, self.lstm_size))
    
    def fit(self, X_train, y_train): # returns model object and fits model on training input 
        train(X_train, y_train, self.model, 1, self.dimensions, max_epochs = 100, debug = self.debug)  # batch_size, dimensions, max_epochs  
        # self.model.fit(X_train, y_train)
        return self.model 
        
    def predict(self, X_test, next_words = 100): 
        self.model.eval()
        words = text.split(' ')
        h_n, c_n = self.model.init_state(len(words))
        for i in range(0, next_words):
            x = torch.tensor([[X.word_to_index[w] for w in words[i:]]])
            y_pred, (h_n, c_n) = self.model(x, (h_n, c_n))
            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim = 0).detach().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            words.append(X.index_to_word[word_index])
        self.preds = words # self.model.predict(X_test) # returns target predictions for testing input 
        return self.preds
    
#     def predict_proba(self, X_test): # REDEFINE for use in ROC_CURVE and looking at probabilities from classifications 
#         self.preds_proba = self.model.predict_proba(X_test) # returns probabilities for target predictions  
#         return self.preds_proba
    
#     def score(self, X_test, y_test):
#         pass # redefine with cross entropy loss function 
#         # self.score = self.model.score(X_test, y_test) # scores model w/ unseen test data 
#         # return self.score
    
#     def accuracy_score(self, y_test, preds, output = True, save = False, path = "", outfile = ""): 
#         self.train_score = self.model.score(self.X, self.y) # X and y are the training/target data 
#         self.test_score = accuracy_score(y_test, preds)
#         report = '{} Train accuracy {:.3f}%'.format(self.model_name, self.train_score * 100) + '\n' \
#             + '{} Test accuracy {:.3f}%'.format(self.model_name, self.test_score * 100) + '\n'
#         self.acc_report = report 
#         if output: 
#             print(report) 
#         if save: 
#             if outfile is "": 
#                 # report.to_csv(path + self.filename + "acc_report.csv", sep = ",", index = False, encoding = 'utf8')
#                 with open(path + self.filename + "_acc_report.txt", "w+", encoding = "utf8") as file: # write byte 
#                     file.write(report)
#             else: 
#                 # report.to_csv(outfile, sep = ",", index = False, encoding = 'utf8')
#                 with open(outfile, "w+", encoding = "utf8") as file: # write byte 
#                     file.write(report) 
#         return self.acc_report 
    
import argparse
from torch.utils.data import DataLoader
# from model import Model
# from dataset import Dataset
    
def train(X, y, model, batch_size, embedding_dim, max_epochs = 10, debug = False):
    if debug: 
        print("Model train") 
    model.train()
    mm = MinMaxScaler()
    ss = StandardScaler()
    X_ss = ss.fit_transform(X)
    y_mm = mm.fit_transform(y) 
    
    X_train = X_ss[:int(0.8 * len(X)), :]
    X_test = X_ss[int(0.8 * len(X)):, :]
    y_train = y_mm[:int(0.8 * len(y)), :] 
    y_test = y_mm[int(0.8 * len(X)):, :]
    # batches = [batch_size] * len(X_train) 
    
    #reshaping to rows, timestamps, features

    X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
    X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 
    
    print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
    print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape)
    
    dataloader = X_train_tensors_final # DataLoader(dataset, batch_size = batch_size)
    criterion = nn.CrossEntropyLoss() # very important 
    learning_rate = 0.001 # 0.005 may be better
    optimizer = optim.Adam(model.parameters(), lr = learning_rate) 
    for epoch in range(max_epochs):
        state_h, state_c = model.init_state(embedding_dim)
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred, (h_n, c_n) = model(x, (h_n, c_n))
            loss = criterion(y_pred.transpose(1, 2), y)
            h_n = h_n.detach()
            c_n = c_n.detach()
            loss.backward()
            optimizer.step()
            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
    

    