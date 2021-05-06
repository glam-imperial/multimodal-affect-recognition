import numpy as np
import pickle
import os
import gensim

from nltk.tokenize import RegexpTokenizer


SEWA_PATH = Path('/path/to/AVEC2017_SEWA')

Transcription_PATH = str(SEWA_PATH / 'transcription_modified') 
Labels_PATH = str(SEWA_PATH / 'labels')
Save_PATH = 'word_embedding'

train_set = [i for i in range(1, 35)]
test_set = [i for i in range(1, 17)]
dev_set = [i for i in range(1, 15)]
inx_dic = {'Train_set' : train_set, 'Test_set' : test_set, 'Devel_set' : dev_set}


class Vocab(object):

    def __init__(self, token2index = None, index2token = None):

        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        if token not in self._token2index:

            index = len(self._index2token)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    def size(self):
        return len(self._token2index)

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            return KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def save(self, filename):
        filename = os.path.join(Save_PATH, filename)
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f)

    def load(cls, filename):
        filename = os.path.join(Save_PATH, filename)
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return token2index, index2token


def savePickle(file, filename):
    path = os.path.join(Save_PATH, filename)
    with open(path, 'wb') as f:
        pickle.dump(file, f)

def loadPickle(filename):
    path = os.path.join(Save_PATH, filename)
    with open(path, 'rb') as f:
        file = pickle.load(f)
    return file


def load_train_corpus(word_embeddings):

    word_vocab = Vocab()
    word_vocab.feed('<unk>')

    tokenizer = RegexpTokenizer(r'\w+')

    for inx in inx_dic['Train_set']:
        filename = 'Train_' + (str(inx) if inx >= 10 else '0' + str(inx)) + '.csv'
        print('reading file: ' + filename)
        with open(os.path.join(Transcription_PATH, filename), 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i, line in enumerate(
                lines):  # fill the frame within a turn duration with the index of words of that duration
            separate_parts = line.split(';')
            text = separate_parts[2]
            text = tokenizer.tokenize(text)
            for word in text:
                word = word.lower()
                word_vocab.feed(word)
    word_vocab.save('vocabulary_training.pkl')
    return word_vocab

def wordEmbeddingMatrix(word_vocab, wordembeddings):

    token2inx, inx2token = word_vocab.load('vocabulary_training.pkl')

    size = len(inx2token)
    EmbeddingMatrix = np.zeros((size, 300))

    for i in range(1, size):
        EmbeddingMatrix[i,:] = wordembeddings[inx2token[i]]
    return EmbeddingMatrix

wordembeddings = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(Save_PATH, 'wiki.de.vec'), binary= False)
word_vocab = load_train_corpus(wordembeddings)
EmbeddingMatrix = wordEmbeddingMatrix(word_vocab, wordembeddings)
savePickle(EmbeddingMatrix, 'Embedding_300_fastText_training.pkl')