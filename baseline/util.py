import numpy as np
import math
import gensim

from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import math

class WordVector(object):
    
    def __init__(self):
        self.model = gensim.models.Word2Vec.load_word2vec_format('apple-text-vector.bin', binary=True)
    
    def word_vector(self, word):
        if word in self.model:
            return self.model[word]
        else:
            return [0.0 for _ in range(200)]

def tf(word, blob):
    return blob.words.count(word)*1.0 / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist)*1.0 / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

def compute_cos_sim(a1,a2):
    num = np.dot(a1,a2)
    if num==0.0:
        return num
    den = np.linalg.norm(a1)*np.linalg.norm(a2)
    return num/den

def bow_vector(text, words, binary=False):
    stop = list(stopwords.words('english'))
    stop+=['how','you','your','to']
    bow_vector = [0.0 for i in range(len(words))]
    for t in text:
        if t.lower() in stop:
            continue
        elif t.isalnum() and t.lower() in words:
            if binary:
                bow_vector[words.index(t.lower())]=1.0
            else:
                bow_vector[words.index(t.lower())]+=1.0
    bow_vector=np.array(bow_vector)
    return bow_vector

def tfidf_bow_vector(text, words, words_file):

    stop = list(stopwords.words('english'))
    stop+=['how','you','your','to']
    bow_vector = [0.0 for i in range(len(words))]
    for t in text:
        if t.lower() in stop:
            continue
        elif t.isalnum() and t.lower() in words_file:
            tf = text.count(t)*1.0/len(text)
            idf = math.log(1000000000.0/(1.0 + len(words_file[t.lower()])))
            assert tf*idf > 0
            bow_vector[words.index(t.lower())] = tf*idf
    bow_vector=np.array(bow_vector)
    return bow_vector

def bow_word_vector(text, wv, binary=False):
    stop = list(stopwords.words('english'))
    stop += ['how','you','your','to']
    word_vector = [0.0 for _ in range(200)]
    
    for t in text:
        t = t.lower()
        if t in stop:
            continue
        elif t.isalnum():
            word_vector = [sum(x) for x in zip(word_vector, wv.word_vector(t))]
    
    return word_vector

# def bigram_vector(text, words, binary=False):


def element_wise_dot(a,b):
    return [x*y for x,y in zip(a,b)]

def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    return continuous_chunk

def get_pos_tag_sentence(sentence):
    word_tokenized_sentence = word_tokenize(sentence)
    return pos_tag(word_tokenized_sentence)
