import re
import pickle
import pandas as pd
import numpy as np
from typing import List
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.csr import csr_matrix

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec as D2V
from gensim.models import KeyedVectors

from transformers import BertTokenizer, BertModel
import torch

EMBEDDINGS_SAVE_PATH = 'Models/Embeddings/'
PRETRAINED_EMBEDDINGS_PATH = 'Data/'

class Embedding:
    def __init__(self, emb_type: str, load_filename='', **kwargs): 
        name_to_class = {'tfidf':    TFIDF(**kwargs), 
                         'doc2vec':  Doc2Vec(**kwargs), 
                         'word2vec': Word2Vec(**kwargs), 
                         'd2v':      Doc2Vec(**kwargs), 
                         'w2v':      Word2Vec(**kwargs),
                         'glove':    GloVe(**kwargs), 
                         'bert':     BERT(**kwargs)}
        
        self.embedding = name_to_class[emb_type]
        
        # init from previously saved embedding
        if load_filename != '':
            self.embedding.__init_from_file__(load_filename)

    def vectorize(self, texts: List[str], unseen=False, **kwargs):
        '''
        Vectorize each text in texts; train models when appropriate (doc2vec).
        
        if unseen is true, infer vectors based on trained models when appropriate
        '''
        return self.embedding.vectorize(texts, unseen=unseen, **kwargs)
    
    def save(self, **kwargs):
        '''
        After running vectorize(), save the sentence vectors/embeddings, 
        or models when appropriate, for the given texts.
        '''
        return self.embedding.save(**kwargs)
    
    def get_filename(self):
        '''
        Get the filename of the saved embeddings/vectors
        '''
        return self.embedding.get_filename()

    

class TFIDF:
    def __init__(self, with_stopwords=False, **kwargs):
        self.with_stopwords = with_stopwords
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
    
    def __init_from_file__(self, filename: str):
        '''
        filename is for the tfidf matrix
        '''
        with_stop = filename.split('_')[-1]
        self.with_stopwords = with_stop == 'withstop'

        with open(EMBEDDINGS_SAVE_PATH + filename, 'rb') as f:
            self.tfidf_matrix = pickle.load(f)
        
        with open(EMBEDDINGS_SAVE_PATH + 'tfidf_vectorizer_' + with_stop, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)


    def vectorize(self, texts: List[str], unseen=False, **kwargs) -> csr_matrix:
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                       max_df = 0.75, min_df=5, 
                                       max_features=10000)
        
        if unseen:
            self.tfidf_matrix = self.tfidf_vectorizer.transform(texts)

        else:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

        return self.tfidf_matrix
    
    def get_filename(self) -> str:
        with_stop = 'withstop' if self.with_stopwords else 'nostop'
        return 'tfidf_matrix_' + with_stop

    def save(self, train_test_split=False, save_test=False):
        '''
        Save the tfidf vectorizer (so vectors can be inferred for unseen texts),
        and the tfidf matrix (the vectors for the given texts).

        params:
            train_test_split: if true, need to append 'train' or 'test' to the filename
            save_test: if true, append 'test' to the tfidf matrix (don't need to save vectorizer in this case)
        '''
        
        with_stop = 'withstop' if self.with_stopwords else 'nostop'
        
        vectorizer_path = EMBEDDINGS_SAVE_PATH + 'tfidf_vectorizer_' + with_stop
        if train_test_split:
            vectorizer_path += '_train'
        
        if self.tfidf_vectorizer is not None and not save_test:
            with open(vectorizer_path, 'wb+') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        matrix_path = EMBEDDINGS_SAVE_PATH + 'tfidf_matrix_' + with_stop
        if train_test_split:
            if save_test:
                matrix_path += '_test'
            else:
                matrix_path += '_train'
        
        if self.tfidf_matrix is not None:
            with open(matrix_path, 'wb+') as f:
                pickle.dump(self.tfidf_matrix, f)



class Doc2Vec:
    def __init__(self, with_stopwords=False, epochs=100, dimensions=300, **kwargs):
        '''
        epochs and dimensions can be set to any (reasonable) value
        '''
        self.with_stopwords = with_stopwords
        self.epochs = epochs
        self.dimensions = dimensions
        self.model = None

    def __init_from_file__(self, filename: str):
        '''
        filename is for d2v model
        '''

        params = filename.split('_')
        
        with_stop = params[-1]
        self.with_stopwords = with_stop == 'withstop'

        self.epochs = int(params[1][:3])
        self.dimensions = int(params[2][:3])

        with open(EMBEDDINGS_SAVE_PATH + filename, 'rb') as f:
            self.model = pickle.load(f)
        
        
    def vectorize(self, texts: List[str], unseen=False, **kwargs) -> List[np.ndarray]:
        # infer vectors of unseen data based on trained model
        if unseen:
            embeddings = [0]*len(texts)
            for i, text in enumerate(texts):
                words = [w for w in text.split(' ')]
                embeddings[i] = self.model.infer_vector(words)

        # get vectors out of a trained model
        else:
            # format as TaggedDocuments
            sents = []
            for id, text in enumerate(texts):
                words = TaggedDocument([w for w in text.split(' ')], [id])
                sents.append(words)

            # train model
            if self.model is None:
                self.model = D2V(documents=sents, min_count=1, window=10, vector_size=self.dimensions, sample=1e-4, negative=5, workers=8)
                self.model.train(corpus_iterable=sents, total_examples=self.model.corpus_count, epochs=self.epochs)

            # get embeddings
            embeddings = self.__get_embeddings_from_model__()
        
        return embeddings
    
    def __get_embeddings_from_model__(self,) -> List[np.ndarray]:
        if self.model is not None:
            return [self.model.dv[i] for i in range(len(self.model.dv))]

    def get_filename(self) -> str:
        with_stop = 'withstop' if self.with_stopwords else 'nostop'
        return 'd2v_' + f'{self.epochs}epochs_{self.dimensions}dim_{with_stop}'

    def save(self, **kwargs):
        '''
        Save the trained doc2vec model.
        '''
        if self.model is not None:
            model_path = EMBEDDINGS_SAVE_PATH + self.get_filename()
            with open(model_path, 'wb+') as f:
                pickle.dump(self.model, f)

  

class Word2Vec:
    def __init__(self, with_stopwords=False, weighting='tfidf', **kwargs):
        '''
        valid weighting: 'tfidf' or 'equal'
        '''
        
        self.pretrained_path = PRETRAINED_EMBEDDINGS_PATH + 'GoogleNews-vectors-negative300.bin.gz'
        self.with_stopwords = with_stopwords
        self.weighting = weighting
        
        self.model = None
        self.avg_vector = None
        #self.embeddings = None

    def __init_from_file__(self, filename: str):
        params = filename.split('_')
        
        with_stop = params[-1]
        self.with_stopwords = with_stop == 'withstop'
        self.weighting = params[1][:-6]

        self.model = KeyedVectors.load_word2vec_format(self.pretrained_path, binary=True)

        with open(EMBEDDINGS_SAVE_PATH + 'w2v_avg_vector', 'rb') as f:
            self.avg_vector = pickle.load(f)
        
        #with open(EMBEDDINGS_SAVE_PATH + filename, 'rb') as f:
        #    self.embeddings = pickle.load(f)
    
    def vectorize(self, texts: List[str], **kwargs) -> List[np.ndarray]:

        # load pretrained embeddings
        if self.model is None:
            self.model = KeyedVectors.load_word2vec_format(self.pretrained_path, binary=True)

        # compute an average vector to be used for unknown words
        try:
            with open(EMBEDDINGS_SAVE_PATH + 'w2v_avg_vector', 'rb') as f:
                self.avg_vector = pickle.load(f)
        except:
            self.avg_vector = sum(vec for vec in self.model.vectors) / len(self.model)
            with open(EMBEDDINGS_SAVE_PATH + 'w2v_avg_vector', 'wb+') as f:
                pickle.dump(self.avg_vector, f)

        # create sentence vectors by specified weighting scheme
        if self.weighting == 'tfidf':
            return self.__tfidf_vectorize__(**kwargs)
        else: # equal weighting/simple avg of each word in sentence
            return self.__equal_vectorize__(texts)
        
    def __tfidf_vectorize__(self, load_train=False, load_test=False, **kwargs) -> List[np.ndarray]:
        '''
        Assumes a tfidf matrix has already been saved (made with the same texts as passed to vectorize()),
        and is ready to load.
        '''
        with_stop = 'withstop' if self.with_stopwords else 'nostop'

        # load tfidf vectorizer and matrix
        vectorizer_path = EMBEDDINGS_SAVE_PATH + f'tfidf_vectorizer_{with_stop}'
        if load_train or load_test:
            vectorizer_path += '_train'
        
        with open(vectorizer_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        matrix_path = EMBEDDINGS_SAVE_PATH + f'tfidf_matrix_{with_stop}'
        if load_train:
            matrix_path += '_train'
        elif load_test:
            matrix_path += '_test'
        
        with open(matrix_path, 'rb') as f:
            tfidf_matrix = pickle.load(f)       

        # get w2v embeddings of the tfidf terms
        terms = tfidf_vectorizer.get_feature_names()
        term_w2v_embeddings = np.zeros((len(terms), 300)) # w2v dimensions are 300
        for i, term in enumerate(terms):
            term_w2v_embeddings[i] = self.model[term] if term in self.model else self.avg_vector       

        # get document vectors weighted by tfidf values
        self.embeddings = tfidf_matrix @ term_w2v_embeddings
        return self.embeddings

    def __equal_vectorize__(self, texts: List[str]) -> List[np.ndarray]:
        
        # get embedding for single sentence
        def get_sent_embedding(sent: str) -> np.ndarray:
            sent_tokens = [word for word in sent.split(' ')]
    
            embedding_sum = np.zeros(300)
            total = 0
            for word in sent_tokens:
                if word in self.model:
                    embedding_sum += self.model[word]
                else:
                    embedding_sum += self.avg_vector
                total += 1
            
            if total == 0:
                embedding = self.avg_vector
            else:
                embedding = embedding_sum / total
            
            return embedding
        
        # return embedding for each sentence
        self.embeddings = [get_sent_embedding(text) for text in texts]
        return self.embeddings
    
    def get_filename(self) -> str:
        with_stop = 'withstop' if self.with_stopwords else 'nostop'
        return f'w2v_{self.weighting}weight_{with_stop}'

    def save(self, **kwargs):
        '''
        Save the sentence embeddings for the given texts
        '''
        if self.embeddings is not None:
            embeddings_path = EMBEDDINGS_SAVE_PATH + self.get_filename()
            with open(embeddings_path, 'wb+') as f:
                pickle.dump(self.embeddings, f)



class GloVe:
    def __init__(self, with_stopwords=False, dimensions=300, weighting='equal', **kwargs):
        '''
        valid dimensions: 50, 100, 200, 300 (make sure you have already downloaded these files)
        valid weighting: 'equal' or 'tfidf'
        '''
        self.with_stopwords = with_stopwords
        self.dimensions = dimensions
        self.weighting = weighting
        self.pretrained_path = PRETRAINED_EMBEDDINGS_PATH + f'glove.6B.{self.dimensions}d.txt'

        self.avg_vector = None
        self.glove_word_embeddings = None
        self.word_to_index = None
        #self.embeddings = None  # training data embeddings

    def __init_from_file__(self, filename: str):
        params = filename.split('_')
        
        with_stop = params[-1]
        self.with_stopwords = with_stop == 'withstop'
        self.dimensions = int(params[1][:3])
        self.weighting = params[2][:-6]

        self.__read_embeddings_from_file__()
        
        #with open(EMBEDDINGS_SAVE_PATH + filename, 'rb') as f:
        #    self.embeddings = pickle.load(f)


    def __read_embeddings_from_file__(self):
        '''
        Read in the pretrained glove word embeddings 
        '''
        # read in word embeddings from file
        self.glove_word_embeddings = []
        self.word_to_index = defaultdict(lambda: -1) # return index -1 on key error 
        with open(self.pretrained_path, 'r') as f:
            for i, line in enumerate(f):
                word, *vector = line.rstrip().split(' ')
                vector = np.array([float(v) for v in vector])
                self.glove_word_embeddings.append(vector)
                self.word_to_index[word] = i

        # create an embedding for unknown words by averaging all glove embeddings in vocab file
        try:
            with open(EMBEDDINGS_SAVE_PATH + f'glove_avg_vector_{self.dimensions}dim', 'rb') as f:
                self.avg_vector = pickle.load(f)
        except:
            self.avg_vector = np.mean(self.glove_word_embeddings, axis=0)
            with open(EMBEDDINGS_SAVE_PATH + f'glove_avg_vector_{self.dimensions}dim', 'wb+') as f:
                pickle.dump(self.avg_vector, f)

        # place at end of glove_embeddings so that when we come across unknown words, 
        # they can be indexed with -1
        self.glove_word_embeddings.append(self.avg_vector)        

    def vectorize(self, texts: List[str], **kwargs) -> List[np.ndarray]:
        if self.glove_word_embeddings is None:
            self.__read_embeddings_from_file__()

        # get sentence vecs based on the specified weighting scheme
        if self.weighting == 'tfidf':
            return self.__tfidf_vectorize__(**kwargs)
        else: # equal weighting/avg each word vector
            return self.__equal_vectorize__(texts)

    def __tfidf_vectorize__(self, load_train=False, load_test=False, **kwargs) -> List[np.ndarray]:
        '''
        Assumes a tfidf matrix has already been saved (made with the same texts as passed to vectorize()),
        and is ready to load.       
        '''
        with_stop = 'withstop' if self.with_stopwords else 'nostop'

        # load tfidf vectorizer and matrix
        vectorizer_path = EMBEDDINGS_SAVE_PATH + f'tfidf_vectorizer_{with_stop}'
        if load_train or load_test:
            vectorizer_path += '_train'
        
        with open(vectorizer_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        matrix_path = EMBEDDINGS_SAVE_PATH + f'tfidf_matrix_{with_stop}'
        if load_train:
            matrix_path += '_train'
        elif load_test:
            matrix_path += '_test'
        
        with open(matrix_path, 'rb') as f:
            tfidf_matrix = pickle.load(f)   
 
        # get the glove embedding for each term in the tfidf matrix
        terms = tfidf_vectorizer.get_feature_names()
        term_glove_embeddings = np.zeros((len(terms), self.dimensions)) 
        for i, term in enumerate(terms):
            term_glove_embeddings[i] = self.glove_word_embeddings[i]       

        # get document vectors weighting by tfidf values
        embeddings = tfidf_matrix @ term_glove_embeddings
        return embeddings

    def __equal_vectorize__(self, texts: List[str]) -> List[np.ndarray]:
        
        # simple average of each word embedding
        def get_sent_embedding(sent: str) -> np.ndarray:
            sent_tokens = [self.word_to_index[word] for word in sent.split(' ')]

            sent_embedding = sum(self.glove_word_embeddings[i] for i in sent_tokens)/len(sent_tokens)
            return sent_embedding     

        self.embeddings = [get_sent_embedding(text) for text in texts]
        return self.embeddings   
    
    def get_filename(self) -> str:
        with_stop = 'withstop' if self.with_stopwords else 'nostop'
        return f'glove_{self.dimensions}dim_{self.weighting}weight_{with_stop}'

    def save(self, **kwargs):
        '''
        Save the sentence embeddings for the given texts 
        '''
        if self.embeddings is not None:
            embeddings_path = EMBEDDINGS_SAVE_PATH + self.get_filename()
            with open(embeddings_path, 'wb+') as f:
                pickle.dump(self.embeddings, f)
                 


class BERT:
    def __init__(self, with_stopwords=False, **kwargs):
        self.with_stopwords = with_stopwords
        self.tokenizer = None
        self.model = None
        self.embeddings = None    # training data embeddings

    def __init_from_file__(self, filename: str):
        params = filename.split('_')
        
        with_stop = params[-1]
        self.with_stopwords = with_stop == 'withstop'

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

    def vectorize(self, texts: List[str], **kwargs) -> torch.Tensor:
        '''
        Based on https://towardsdatascience.com/build-a-bert-sci-kit-transformer-59d60ddd54a5
        '''
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = BertModel.from_pretrained("bert-base-uncased")

        with torch.no_grad():
            self.embeddings = torch.stack([self.__get_sent_embedding__(text) for text in texts])
        
        return self.embeddings

    def __get_sent_embedding__(self, sent: str) -> torch.Tensor:
        '''
        Get bert embedding for one sentence.
        '''
        tokenized = self.tokenizer.encode_plus(sent, add_special_tokens=True)['input_ids']
        attention = [1] * len(tokenized)

        tokenized = torch.tensor(tokenized).unsqueeze(0)
        attention = torch.tensor(attention).unsqueeze(0)

        embedding = self.model(tokenized, attention)
        return embedding[0][:, 0, :].squeeze()
    
    def get_filename(self) -> str:
        with_stop = 'withstop' if self.with_stopwords else 'nostop'
        return f'bert_{with_stop}'

    def save(self, **kwargs):
        '''
        Save bert sentence embeddings for given texts
        '''
        if self.embeddings is not None:
            embeddings_path = EMBEDDINGS_SAVE_PATH + self.get_filename()
            with open(embeddings_path, 'wb+') as f:
                pickle.dump(self.embeddings, f)


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

    e = Embedding('word2vec', with_stopwords=True, weighting='tfidf', epochs=50, dimensions=50)
    
    vectors = e.vectorize(tweets, load_test=True)
    print(vectors[0])

    #e.save()