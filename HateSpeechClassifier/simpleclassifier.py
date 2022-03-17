import pickle
import re
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def utils_preprocess_text(text: str, flg_stemm=False, flg_lemm=False, lst_stopwords=None) -> str:
    '''
    Code from https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
    Preprocess a string.
    :parameter
        :param text: string - text to process
        :param lst_stopwords: list - list of stopwords to remove
        :param flg_stemm: bool - whether stemming is to be applied
        :param flg_lemm: bool - whether lemmitization is to be applied
    :return
        cleaned text
    '''
    
    # clean (convert to lowercase and remove punctuations and   
    # characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    # Tokenize (convert from string to list)
    lst_text = text.split()
    
    # remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    # Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    # Lemmatization (convert the word into root word)
    if flg_lemm == True:
        lem = WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    # back to string from list
    text = " ".join(lst_text)
    return text

def classify(texts: List[str], 
            vectorizer_path='Models/Embeddings/tfidf_vectorizer_nostop', 
            model_path='Models/MultiClassifier/LR_tfidf_matrix_nostop'
            ):

    with open(vectorizer_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open(model_path, 'rb') as f:
        lr_model = pickle.load(f)

    texts = [utils_preprocess_text(text) for text in texts]
    vectorized_texts = tfidf_vectorizer.transform(texts)

    # has probabilities for each class
    probabilities = lr_model.predict_proba(vectorized_texts)
    
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
    texts = ["Happy birthday bird! @codyflitcraft",
            "Haha, Yankees fans are bent out of shape about the Hu27le graphic",
            "Maybe the UN could talk to those asian and african nations responsible for 90%+ of the pollution in the oceans' instead of insisting on this bullshit about climate change. ",
            "Jews are a separate ethnicity, a danger to national security, and should be banned.",
            "Gay men are a danger to children.",
            "We are being invaded, We must fight until we will vanquish immigrants."]
    classes, probs = classify(texts)
    
    for i, text in enumerate(texts):
        print(text)
        print(f'  Target = {classes[i]}, probability = {probs[i]}')
