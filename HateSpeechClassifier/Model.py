import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from xgboost import XGBClassifier

MODELS_PATH = '../HateSpeechClassifier/Models/MultiClassifier/'

class Model:
    def __init__(self, model_name: str, load_filename='', **kwargs):
        self.model_name = model_name
        
        if load_filename != '':
            self.__init_from_file__(load_filename)
        else:
            name_to_class = {'LR': LogisticRegression(max_iter=2000),
                             'RF': RandomForestClassifier(**kwargs),
                             'DT': DecisionTreeClassifier(**kwargs),
                             'SVM': SVC(probability=True, **kwargs),
                             'XGB': XGBClassifier(**kwargs)}

            self.model = name_to_class[model_name]
    
    def __init_from_file__(self, filename: str):
        with open(MODELS_PATH + filename, 'rb') as f:
            self.model = pickle.load(f)
    
    def save(self, embedding_name: str):
        path = MODELS_PATH + self.model_name + '_' + embedding_name
        with open(path, 'wb+') as f:
            pickle.dump(self.model, f)