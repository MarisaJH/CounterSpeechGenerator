import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

MODELS_PATH = 'Models/MultiClassifier/'

class Model:
    def __init__(self, model_name: str, **kwargs):
        name_to_class = {'LR': LogisticRegression(**kwargs),
                         'RF': RandomForestClassifier()}
        self.model_name = model_name
        self.model = name_to_class[model_name]
    
    def save(self, embedding_name: str):
        path = MODELS_PATH + self.model_name + '_' + embedding_name
        with open(path, 'wb+') as f:
            pickle.dump(self.model, f)