import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

MODELS_PATH = 'Models/MultiClassifier/'

class Model:
    def __init__(self, model_name: str, load_filename='', **kwargs):
        self.model_name = model_name
        
        if load_filename != '':
            self.__init_from_file__(load_filename)
        else:
            name_to_class = {'LR': LogisticRegression(**kwargs),
                             'RF': RandomForestClassifier()}

            self.model = name_to_class[model_name]
    
    def __init_from_file__(self, filename: str):
        with open(MODELS_PATH + filename, 'rb') as f:
            self.model = pickle.load(f)
    
    def save(self, embedding_name: str):
        path = MODELS_PATH + self.model_name + '_' + embedding_name
        with open(path, 'wb+') as f:
            pickle.dump(self.model, f)