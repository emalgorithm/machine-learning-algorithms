from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import numpy as np

tknzr = TweetTokenizer()
sws = stopwords.words('english')


def preprocess_data():
    idx = 0
    X = []
    y = []
    word_to_index = {}
    words_vocabulary = []
    
    
    with open('../data/reviews_sentiment/amazon_cells_labelled.txt', 'r') as f:
        for l in f:
            sentence, sentiment = l.split('\t')
            sentiment = int(sentiment)
            y.append(sentiment)

            words = tknzr.tokenize(sentence)
            # Remove stopwords does not improve accuracy but makes the model faster as we have less words to process
            words = [word for word in words if word not in sws]
            X.append(words)
            
            for word in words:
                if word not in word_to_index:
                    word_to_index[word] = idx
                    words_vocabulary.append(word)
                    idx += 1
    
    return X, y, word_to_index, words_vocabulary

def text_to_features(X, words_vocabulary):
    X_feat = []
    for x in X:
        x_feat = [1 if word in x else 0 for word in words_vocabulary]
        X_feat.append(x_feat)
    
    return X_feat

