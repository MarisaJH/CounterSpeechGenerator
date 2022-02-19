import pickle
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def classify(texts: List[str], 
            vectorizer_path='Models/Embeddings/tfidf_vectorizer', 
            model_path='Models/MultiClassifier/LR_model_tfidf'
            ):

    with open(vectorizer_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open(model_path, 'rb') as f:
        lr_model = pickle.load(f)

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
            "Maybe the UN could talk to those asian and african nations responsible for 90%+ of the pollution in the oceans' instead of insisting on this bullshit about climate change. "]
    classes, probs = classify(texts)
    
    for i, text in enumerate(texts):
        print(text)
        print(f'  Target = {classes[i]}, probability = {probs[i]}')
