import json, pickle, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, KFold, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_roc_curve

from typing import List, Dict, Tuple
from collections import defaultdict

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from Embedding import Embedding, utils_preprocess_text, EMBEDDINGS_SAVE_PATH 
from Model import Model, MODELS_PATH 

REPORTS_PATH = 'Reports/'
CLASSIFS_PATH = 'Classifications/' # Use for classifications made using classify or similar function to test models 

SEED = 42 # Mo - maybe we should  try and use the same seed number throughout the code? - was 100
TARGET_TYPES = ['Disabled', 'Jews', 'LGBT+', 'Migrants', 'Muslims', 'POC', 'Women', 'Other/Mixed', 'None']

def run_tests_kfold(texts: List[str], class_labels: List[int],
                    embedding_types = ['tfidf', 'word2vec', 'doc2vec', 'glove', 'bert'],
                    model_types = ['LR', 'RF', 'NB', 'DT', 'XGB', 'SVC'],
                    with_stopwords = False,
                    weighting_type ='equal',  
                    dimensions = 300,
                    scoring = ['accuracy', 'balanced_accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovr'],
                    debug = True
                    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    '''
    Based on: https://towardsdatascience.com/quickly-test-multiple-models-a98477476f0
    
    Run kfold cross validation on multiple embedding/model combos. 

    params:
        texts: list of strings (posts), already stripped of punctuation/symbols
        class_labels: class labels
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

    X_train_text, X_test_text, y_train, y_test = train_test_split(texts, class_labels, random_state = SEED, test_size = 0.2)

    dfs = [] # save results for each scoring metric for each k in the kfold
    confusions = {} # save one confusion matrix for each embedding/model combo
    for embedding_type in embedding_types:
        if debug:
            print() 
            print('----------------------------------------')
            print(embedding_type)

        embedding = Embedding(embedding_type, 
                              with_stopwords = with_stopwords, 
                              weighting = weighting_type,
                              dimensions = dimensions)
        
        # vectorize train and test set
        X_train = embedding.vectorize(X_train_text, load_train=False)# True)

        # might need to save tfidf vectorizer and matrix for later use
        # if embedding_type == 'tfidf' and weighting_type == 'tfidf':
        embedding.save(train_test_split=True)

        X_test = embedding.vectorize(X_test_text, unseen=True, load_test=False)#True)
        
        # if embedding_type == 'tfidf' and weighting_type == 'tfidf':
        embedding.save(train_test_split = True, save_test = True)

        for model_type in model_types:
            if debug:
                print('--------- ' + model_type +  ' -----------')
                # print('  ' + model_type)
                
            if model_type == 'NB': 
                # vectorize train and test set for NB 
                X_train = embedding.vectorize(X_train_text, load_train=False).toarray() # True)
                embedding.save(train_test_split=True)

                X_test = embedding.vectorize(X_test_text, unseen=True, load_test=False).toarray() #True)
                embedding.save(train_test_split=True, save_test=True)
            
            model_params = {'max_iter' : 2000 }
            # run model selection, with k-fold cross validation
            model = Model(model_type)
            
            kfold = KFold(n_splits = 5, shuffle = True, random_state = SEED)
            cv_results = cross_validate(model.model, X_train, y_train, cv = kfold, scoring = scoring)
            clf = model.model.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # add results to dataframe
            model_name = model_type + '_' + embedding.get_filename()

            this_df = pd.DataFrame(cv_results)
            this_df['model'] = model_name
            dfs.append(this_df)
            
            # save confusion
            confusion = confusion_matrix(y_test, y_pred)
            confusions[model_name] = confusion

            if debug:
                print(classification_report(y_test, y_pred, target_names = TARGET_TYPES))  

    final_df = pd.concat(dfs, ignore_index = True)
    return final_df, confusions

def train_and_save(texts: List[str], class_labels: List[int],
                   embedding_types = ['tfidf', 'word2vec', 'doc2vec', 'glove', 'bert'],
                   model_types = ['LR', 'RF', 'NB', 'DT', 'XGB', 'SVC'],
                   with_stopwords = False,
                   weighting_type = 'equal',  
                   dimensions = 300,
                   scoring = ['accuracy', 'balanced_accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovr'], 
                   train_test = True, 
                   debug = True):
    '''
    Train multiple combinations of embeddings and models on an entire training dataset,
    and save the embeddings and models.

    params:
        texts: list of strings (posts), already stripped of punctuation/symbols
        class_labels: class labels
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
    
    # Add scoring for hyperparameter tuning 
    # Final cross validation dictionary/dataframe comparison 
    model_dfs = [] #  {} 
    
    # remove stopwords if necessary
    if not with_stopwords:
        stop_words = stopwords.words('english')
        texts = [t for t in texts if not t in stop_words]
    
    if train_test: 
        X_train_text, X_test_text, y_train, y_test = train_test_split(texts, class_labels, random_state = SEED, test_size = 0.2)
    else: 
        X_train_text = texts # vectorize texts for 100% data for train
        y_train = class_labels # class_labels for 100% data for train
        
    for embedding_type in embedding_types:
        if debug:
            print('----------------------------------------')
            print(embedding_type)

        # get embeddings for input text
        embedding = Embedding(embedding_type, 
                              with_stopwords = with_stopwords, 
                              weighting = weighting_type,
                              dimensions = dimensions)
        
        embedding_filename = embedding.get_filename()
        
        # vectorize train and test set
        if (embedding_type == 'tfidf'): 
            X = embedding.vectorize(X_train_text, load_train = True) 
            y = y_train 

            if train_test:
                X_test = embedding.vectorize(X_test_text, unseen=True, load_test = True)
        else: 
            X = Embedding.vectorize(X_train_text, load_train = False) 
            y = y_train  

            # might need to save tfidf vectorizer and matrix for later use
            embedding.save(train_test_split=True)

            if train_test: 
                X_test = embedding.vectorize(X_test_text, unseen=True, load_test = False)
                embedding.save(train_test_split=True, save_test=True)
                    
        # model_num = 0
        debug = debug 
        output = True 
        save = True # False 
        num = True
        cross_validate = True # whether to fine tune the hyper parameters of the model with a cross validator 
        outfile = ""
           
        for model_type in model_types:
            if model_type != "NB" or model_type != "Naive Bayes": 
                X = embedding.vectorize(X_train_text, load_train=False) 
                y = y_train
                if train_test: 
                    X_test = embedding.vectorize(X_test_text, unseen=True, load_test=False)
                    embedding.save(train_test_split=True, save_test=True)
            else: # for Naive Bayes 
                X = X.toarray()
                if train_test: 
                    X_test = X_test.toarray() 
                
            model_name = model_type + "_" + embedding_filename 
            model_params = {
                           'random_state' : SEED
                           }

            if debug:
                print('----------------------------------------')
                print(model_type)
                print('----------------------------------------')
                print("Filename ")
                print("", model_name) 
                
            if (model_type == "LR" or model_type == "Logistic Regression"): 
                if cross_validate:                     
                    model_params['max_iter'] = 1000 # control number of iterations for regression convergence 
                    solvers = ['newton-cg', 'lbfgs', 'liblinear']
                    penalty = ['l2']
                    c_values = [100, 10, 1.0, 0.1, 0.01]
                    # define grid search
                    param_grid = {
                                  'penalty': penalty,
                                  'C': c_values,
                                  'solver': solvers, # penalty, C, solver is order 
                                 }
                    cv = GridSearchCV # almost always uses StratifiedKFold with 5 splits 
                    cv_params = {'param_grid': param_grid, 
                                 'scoring': scoring, 
                                 'verbose': 3
                                } # [param_grid, {'verbose': 3}]
            elif (model_type == "RF" or model_type == "Random Forest"):  # do with randomized search 
                if cross_validate: 
                    cv = RandomizedSearchCV # almost always uses StratifiedKFold with 5 splits 
                    # Number of trees in random forest
                    n_estimators = [int(x) for x in np.linspace(start = 800, stop = 2000, num = 3)] # [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
                    # Number of features to consider at every split
                    max_features = ['auto', 'sqrt']
                    # Maximum number of levels in tree
                    max_depth = [int(x) for x in np.linspace(10, 110, num = 3)] # [int(x) for x in np.linspace(10, 110, num = 11)]
                    max_depth.append(None)
                    # Minimum number of samples required to split a node
                    min_samples_split = [2, 5, 10]
                    # Minimum number of samples required at each leaf node
                    min_samples_leaf = [1, 2, 4]
                    bootstrap = [True, False]
                    # param_grid = {'base_estimator__max_depth': [2, 4, 6, 8]}
                    # Create the param grid
                    param_grid = { 
                                   'n_estimators': n_estimators,
                                   'max_features': max_features,
                                   'max_depth': max_depth,
                                   'min_samples_split': min_samples_split,
                                   'min_samples_leaf': min_samples_leaf,
                                   'bootstrap': bootstrap # Method of selecting samples for training each tree
                                 }
                    cv_params = {'param_distributions': param_grid, 
                                 'scoring': scoring, 
                                 'verbose': 2
                                } # [param_grid, {'verbose': 3}]
            elif (model_type == "DT" or model_type == "Decision Trees"): 
                if cross_validate: 
                    # Number of features to consider at every split
                    max_features = ['auto', 'sqrt']
                    # Maximum number of levels in tree
                    max_depth = [int(x) for x in np.linspace(10, 110, num = 3)] # [int(x) for x in np.linspace(10, 110, num = 11)]
                    max_depth.append(None)
                    # Minimum number of samples required to split a node
                    min_samples_split = [2, 5, 10]
                    # Minimum number of samples required at each leaf node
                    min_samples_leaf = [1, 2, 4]
                    # choose function to measure quality of node split 
                    criterion = ["gini", "entropy"]
                    param_grid = {'criterion' : criterion,
                                  'max_features': max_features,
                                  'max_depth': max_depth,
                                  'min_samples_split': min_samples_split,
                                  'min_samples_leaf': min_samples_leaf,
                                 }
                    cv = RandomizedSearchCV # almost always uses StratifiedKFold with 5 splits 
                    cv_params = {'param_distributions': param_grid, 
                                 'scoring': scoring, 
                                 'verbose': 2
                                } # [param_grid, {'verbose': 3}]
            elif (model_type == "XGB" or model_type == "XGBoost"): 
                if cross_validate: 
                    ''' Define search space for model parameters ''' 
                    param_grid = {'objective': ['reg:squarederror', 'reg:squaredlogerror'],
                                  'max_depth': [3, 6, 10],
                                  'learning_rate': [0.01, 0.05, 0.1],
                                  'n_estimators': [100, 500, 1000],
                                  'colsample_bytree': [0.3, 0.7]
                                  } 
                    cv = RandomizedSearchCV
                    cv_params = {'param_distributions': param_grid,  # 'param_grid' : param_grid, use grid with gridsearch 
                                 'scoring': scoring, 
                                 'verbose': 2
                                } # [param_grid, {'verbose': 3}]
            elif (model_type == "SVC" or model_type == "SVM"):  # Support Vector Machine
                if cross_validate: 
                    param_grid = { 
                                   'C': [0.1, 1, 10, 100], 
                                   'gamma': [1 , 0.1, 0.01, 0.001]
                                 } 
                    cv = GridSearchCV # RandomizedSearchCV
                    cv_params = {'param_grid': param_grid, 
                                 'scoring': scoring, 
                                 # 'refit': True, 
                                 'verbose': 2
                                } # [param_grid, {'verbose': 3}]
            elif (model_type == "NB" or model_type == "Naive Bayes"):
                X = X.toarray()
                X_test = X_test.toarray() 
                
                y_class = pd.DataFrame(y, columns = ["CLASS"])
                priors = y_class.groupby('CLASS').size().div(len(y)) # get prior probs of the train/target classes 
                priors = list(priors) 
                model_params = {'priors': priors }
                if cross_validate: 
                    cv = GridSearchCV 
                    param_grid = {'var_smoothing': np.logspace(0, -9, num = 100)
                # np.<a onclick="parent.postMessage({'referent':'.numpy.logspace'}, '*')">logspace(0,-9, num=100)}
                                 }     
                    cv_params = {'param_grid': param_grid, 
                                 'scoring': scoring, 
                                 'verbose': 3
                                } # [param_grid, {'verbose': 3}]
            
            if cross_validate:
                model = Model(model_type, X = X, y = y, debug = debug, model_params = model_params, filename = model_name, 
                              output = True, save = True, path = REPORTS_PATH, 
                              cv = cv, cv_params = cv_params)   # max_iter=2000)
                # model.fit(X, y)  Called when Model called with a cross validator so unnecessary 
                model.report_cv_results(output = output, save = save, path = REPORTS_PATH)
                best_model, best_model_score, best_model_params = model.get_params() 
                models_index = [np.array(['Model'] * len(models)),
                        np.array(models)]
                scores_index = [np.array(['Model Score'] * len(models)),
                                np.array([0.0] * len(models)), 
                               ]         
                params_index = [np.array(['Model Params'] * len(best_model_params.keys)),
                                np.array(best_model_params.keys),
                               ]

                params_df = pd.concat([pd.DataFrame(best_model, index = models_index), 
                               pd.DataFrame(best_model_score, index = scores_index)]) # .transpose()
                if debug: 
                    print("Model parameters: ") 
                    print(params_df)

                model_params_df = pd.DataFrame(best_model_params, index = params_index) # (pd.DataFrame.from_dict(self.model_params))# .transpose()
                model_compare_df = pd.concat([params_df, model_params_df])        
                model_dfs.append(model_compare_df) 
                # model_dfs[model_name] = model_compare_df
                
                if output:
                    print("Model paramters comparison: ") 
                    print(model_compare_df)
                if save: 
                    model_compare_df.to_csv(REPORTS_PATH + model_name + "_comparison_report.csv", sep = ",", index = False, encoding = 'utf8')   
            elif (model_type == "LSTM"):
                model = NeuralNet(model_type, X = X, y = y, debug = debug, 
                     model_params = model_params, filename = model_name) #max_iter=2000) 
                # LSTM(X, y, debug = debug) #max_iter=2000)
                #     mm = MinMaxScaler()
                #     ss = StandardScaler()
                #     X_ss = ss.fit_transform(X)
                #     y_mm = mm.fit_transform(y) 

                #     X_train = X_ss[:int(0.8 * len(X)), :]
                #     X_test = X_ss[int(0.8 * len(X)):, :]
                #     y_train = y_mm[:int(0.8 * len(y)), :] 
                #     y_test = y_mm[int(0.8 * len(X)):, :]
                #     batches = [batch_size] * len(X) 
                #     train(model)    
                model.fit(X, y)
                break 
            else: 
                model = Model(model_type, X = X, y = y, debug = False, model_params = model_params, filename = model_name) #max_iter=2000) 
                model.fit(X, y) 
                
            y_preds = model.predict(X_test)
            # preds_report = model.predictions_report(y_test, y_preds, output = output, save = save, 
                                                    # path = REPORTS_PATH) 
            cls_report = model.classification_report(y_test, y_preds, output = output, save = save, 
                                                     path = REPORTS_PATH) #, path = reports_path)
            acc_report = model.accuracy_score(y_test, y_preds, path = REPORTS_PATH)
            conf_m = model.confusion_matrix(y_test, y_preds, output = output, save = save, 
                                            path = REPORTS_PATH) #, path = reports_path)
            model.heatmap(save = save, num = num, path = REPORTS_PATH) # path = reports_path)
            model.roc_curve(X_test, y_test, save = save, path = REPORTS_PATH) # path = reports_path)
            # model.report_results(output = output, save = save, path = REPORTS_PATH) # path = reports_path) 
            model.save_model(MODELS_PATH)
            
            print("Cross Validation Evaluation: ") 
            print("Cross validator: ", model.CV) 
            print("Old estimator: ", model.untuned_model) 
            print("Best estimator: ", model.model)
            print("Best parameters: ", model.model_params) 
            # print("Cross_validation report comparison: ")
            # print(model.cv_report) 
            
    # Save final comparison dataframes
    final_models = pd.concat(model_dfs, ignore_index = True) # pd.DataFrame(model_dfs) # , ignore_index = True)  
    if output: 
        print(final_models)
    if save: 
        final_models.to_csv(REPORTS_PATH + "_models_comparison_report.csv", sep = ",", index = False, encoding = 'utf8')            
    
def classify(texts: List[str], model_path: str, debug = False) -> Tuple[List[str], List[float]]:
    '''
    Classify unseen text

    params:
        texts: list of strings (social media posts)
        model_path: string, path to saved model to use for classification

    returns: parallel lists of predicted classes, predicted probabilites
    '''
    # preprocess input
    suffix = model_path.split('.')[-1]
    filename = model_path.split('/')[-1]
    model_name, *embedding_params = filename.split('_')
    embedding_filename = '_'.join(param for param in embedding_params)
    embedding_filename = embedding_filename[:-10] if "_model." in filename else print(filename) # name without suffix 
    embedding_type = embedding_params[0]

    if filename.endswith('nostop'):
        stop_words = stopwords.words('english')
    else:
        stop_words = None
    
    texts = [utils_preprocess_text(text, lst_stopwords = stop_words) for text in texts]
    if debug: 
        print(embedding_filename) 
        print(embedding_filename in os.listdir(EMBEDDINGS_SAVE_PATH))
        print(os.listdir(EMBEDDINGS_SAVE_PATH)) 
    # transform input text into embedding
    if (embedding_filename in os.listdir(EMBEDDINGS_SAVE_PATH)): # skip if .ipynb_checkpoints or embedding is no longer in directory 
        embedding = Embedding(embedding_type, load_filename = embedding_filename)
        vectorized_texts = embedding.vectorize(texts, unseen = True)

        # predict
        ''' includes .pkl for models - will change to match for both embeddings and models later '''
        full_model_path = MODELS_PATH + model_path if suffix in model_path else MODELS_PATH + model_name + '_' + embedding_filename 
        model = Model(model_name, load_filename = full_model_path) # , max_iter=2000) 
        probabilities = model.predict_proba(vectorized_texts)

        # return classes with highest probs for each text
        predicted_classes = [0] * len(texts)
        predicted_probs = [0] * len(texts)

        for i, probs in enumerate(probabilities):
            max_prob = max(probs)
            max_class = np.where(probs == max_prob)[0][0]
            predicted_classes[i] = TARGET_TYPES[max_class]
            predicted_probs[i] = max_prob

        return predicted_classes, predicted_probs
    return None, None 

if __name__ == '__main__':
  
    combined_data_path = 'Data/combined_data.csv'
    # Make True if you want to save processed data to file and load when ready for next run time - make False if processed_data file is already in given PATH
    save_data = False   
    # Make True if you need to realign new texts with new class labels (a new data)
    dict_data = False
    
    if dict_data: 
        with open(combined_data_path, 'r', encoding='utf8') as f:
            combined = pd.read_csv(f).to_dict()

        tweets = [] # these arrays are aligned by index; tweet[i] has label[i]
        class_labels = []
        for tweet, class_label in zip(combined['HATE_SPEECH'].values(), combined['CLASS'].values()):

            tweet = utils_preprocess_text(tweet) #, lst_stopwords = stop_words)

            tweets.append(tweet)
            class_labels.append(class_label)
            
        processed_data = pd.DataFrame({'HATE_SPEECH': tweets, 
                                       'CLASS': class_labels})
    else: 
        processed_data = pd.read_csv('Data/processed_combined_data.csv', index_col = False) # ignore index column 
        # processed_data.name = 'Processed Data'
     
    if processed_data is not None and save_data:
        processed_data.to_csv('Data/processed_combined_data.csv', index = False)        
            
    # print(processed_data)
    tweets = processed_data['HATE_SPEECH']
    class_labels = processed_data['CLASS']
                                  
    # train on the above data and save multiple combinations of embeddings/models 
    # train_and_save(tweets, class_labels, debug = False, embedding_types = [
                                                                           # 'tfidf', 
                                                        #                    'word2vec',
                                                        #                    'doc2vec',
                                                        #                    'glove',
                                                        #                    'bert'
                                                        #                    ],
                                                        # model_types = [
                                                        #                'LR',
                                                        #                # 'RF',
                                                        #                # 'NB', 
                                                        #                'XGB',
                                                                      # ], 
                    # scoring = None) # when scoring is not None, refit must be False 
    
    '''
    run_tests_kfold params:
        texts: list of strings (posts), already stripped of punctuation/symbols
        class_labels: class labels
        embedding_types: which embeddings to use; can pass in less than the default
        model_types: which models to use; can pass in less than default
        with_stopwords: whether to remove stopwords from input
        weighting_type: how to weight word vectors for word2vec and glove. Other option is 'tfidf'
        dimensions: for glove and doc2vec. Other options are 50, 100, 200 for glove, any for doc2vec
    ''' 
    # df, confusion = run_tests_kfold(tweets, class_labels, debug = True, embedding_types=['tfidf'], model_types=['LR'])
    # print(df)
    # for key, val in confusion.items(): 
        # print(key)
        # print(val)
        # print() 
    
    texts = ["Happy birthday bird! @codyflitcraft",
        "Haha, Yankees fans are bent out of shape about the Hu27le graphic",
        "Maybe the UN could talk to those asian and african nations responsible for 90%+ of the pollution in the oceans' instead of insisting on this bullshit about climate change. ",
        "Jews are a separate ethnicity, a danger to national security, and should be banned.",
        "Gay men are a danger to children.",
        "We are being invaded, We must fight until we will vanquish immigrants."]
    
    # classify the above texts using every model currently saved in the models directory
    ''' Uncomment when done tuning models with cross validation '''
    
    # print(os.listdir(MODELS_PATH))
    save_comparisons = '' 
    for model_filename in os.listdir(MODELS_PATH):
        if not model_filename.startswith('.'): # to remove '.ipynb_checkpoints'
            model_name = model_filename.split('/')[-1]
            save_comparisons += model_name + '\n' + '----------------------------------------\n\n' 
            # print('----------------------------------------')
            # print(model_name + ':\n')
            classes, probs = classify(texts, model_name)
            if classes != None and probs != None:  # If we skip an embedding that has been removed 
                for i, text in enumerate(texts):
                    # print(text)
                    # print(f'  Target = {classes[i]}, probability = {probs[i]}')
                    save_comparisons += text + '\n' + f' Target = {classes[i]}, probability = {probs[i]}\n\n'
    print(save_comparisons) 
    with open(CLASSIFS_PATH + "save_comparisons.txt", "w+", encoding = "utf8") as f: 
        f.write(save_comparisons)
    
