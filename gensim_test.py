import gensim
from gensim import corpora
from pprint import pprint

text = ["""In computer science, artificial intelligence (AI),
             sometimes called machine intelligence, is intelligence
             demonstrated by machines, in contrast to the natural intelligence
             displayed by humans and animals. Computer science defines
             AI research as the study of intelligent agents: any device that
             perceives its environment and takes actions that maximize its chance
             of successfully achieving its goals."""]

tokens = [[token for token in sentence.split()] for sentence in text]
gensim_dictionary = corpora.Dictionary(tokens)

print("The dictionary has: " +str(len(gensim_dictionary)) + " tokens")

for k, v in gensim_dictionary.token2id.items():
    print(f'{k:{15}} {v:{10}}')

print(gensim_dictionary.token2id["study"])
print(list(gensim_dictionary.token2id.keys())[list(gensim_dictionary.token2id.values()).index(40)])
print(gensim_dictionary.token2id)

text = ["""Colloquially, the term "artificial intelligence" is used to
           describe machines that mimic "cognitive" functions that humans
           associate with other human minds, such as "learning" and "problem solving"""]

tokens = [[token for token in sentence.split()] for sentence in text]
gensim_dictionary.add_documents(tokens)

print("The dictionary has: " + str(len(gensim_dictionary)) + " tokens")
print(gensim_dictionary.token2id)

from gensim.utils import simple_preprocess
from smart_open import smart_open
import os

gensim_dictionary = corpora.Dictionary(simple_preprocess(sentence, deacc=True) for sentence in open(r'gensim_test/sample1.txt', encoding='utf-8'))

print(gensim_dictionary.token2id)

class ReturnTokens(object):
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def __iter__(self):
        for file_name in os.listdir(self.dir_path):
            for sentence in open(os.path.join(self.dir_path, file_name), encoding='utf-8'):
                yield simple_preprocess(sentence)

path_to_text_directory = r"D:\python\nlp_test\gensim_test"
gensim_dictionary = corpora.Dictionary(ReturnTokens(path_to_text_directory))

print(gensim_dictionary.token2id)

text = ["""In computer science, artificial intelligence (AI),
           sometimes called machine intelligence, is intelligence
           demonstrated by machines, in contrast to the natural intelligence
           displayed by humans and animals. Computer science defines
           AI research as the study of intelligent agents: any device that
           perceives its environment and takes actions that maximize its chance
           of successfully achieving its goals."""]

tokens = [[token for token in sentence.split()] for sentence in text]

gensim_dictionary = corpora.Dictionary()
gensim_corpus = [gensim_dictionary.doc2bow(token, allow_update=True) for token in tokens]

print(gensim_corpus)

word_frequencies = [[(gensim_dictionary[id], frequence) for id, frequence in couple] for couple in gensim_corpus]
print(word_frequencies)

tokens = [simple_preprocess(sentence, deacc=True) for sentence in open(r'gensim_test/sample1.txt', encoding='utf-8')]

gensim_dictionary = corpora.Dictionary()
gensim_corpus = [gensim_dictionary.doc2bow(token, allow_update=True) for token in tokens]
word_frequencies = [[(gensim_dictionary[id], frequence) for id, frequence in couple] for couple in gensim_corpus]

print(word_frequencies)

class ReturnTokens(object):
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def __iter__(self):
        for file_name in os.listdir(self.dir_path):
            for sentence in open(os.path.join(self.dir_path, file_name), encoding='utf-8'):
                yield simple_preprocess(sentence)

path_to_text_directory = r"D:\python\nlp_test\gensim_test"

gensim_dictionary = corpora.Dictionary()
gensim_corpus = [gensim_dictionary.doc2bow(token, allow_update=True) for token in ReturnTokens(path_to_text_directory)]
word_frequencies = [[(gensim_dictionary[id], frequence) for id, frequence in couple] for couple in gensim_corpus]

print(word_frequencies)

text = ["I like to play Football",
       "Football is the best game",
       "Which game do you like to play ?"]

tokens = [[token for token in sentence.split()] for sentence in text]

gensim_dictionary = corpora.Dictionary()
gensim_corpus = [gensim_dictionary.doc2bow(token, allow_update=True) for token in tokens]

from gensim import models
import numpy as np

tfidf = models.TfidfModel(gensim_corpus, smartirs='ntc')

for sent in tfidf[gensim_corpus]:
    print([[gensim_dictionary[id], np.around(frequency, decimals=2)] for id, frequency in sent])

import gensim.downloader as api

w2v_embedding = api.load("glove-wiki-gigaword-100")
print(w2v_embedding.most_similar('toyota'))