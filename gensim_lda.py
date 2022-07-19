import wikipedia
import nltk

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

global_warming = wikipedia.page("Global Warming", auto_suggest=False)
artificial_intelligence = wikipedia.page("Artificial Intelligence", auto_suggest=False)
mona_lisa = wikipedia.page("Mona Lisa", auto_suggest=False)
eiffel_tower = wikipedia.page("Eiffel Tower", auto_suggest=False)

corpus = [global_warming.content, artificial_intelligence.content, mona_lisa.content, eiffel_tower.content]

import re
from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

def preprocess_text(document):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        tokens = document.split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in en_stop]
        tokens = [word for word in tokens if len(word)  > 5]

        return tokens

processed_data = [];
for doc in corpus:
    tokens = preprocess_text(doc)
    processed_data.append(tokens)

from gensim import corpora

gensim_dictionary = corpora.Dictionary(processed_data)
gensim_corpus = [gensim_dictionary.doc2bow(token, allow_update=True) for token in processed_data]

import pickle

pickle.dump(gensim_corpus, open('gensim_corpus_corpus.pkl', 'wb'))
gensim_dictionary.save('gensim_dictionary.gensim')

import gensim

lda_model = gensim.models.ldamodel.LdaModel(gensim_corpus, num_topics=4, id2word=gensim_dictionary, passes=20)
lda_model.save('gensim_model.gensim')
topics = lda_model.print_topics(num_words=10)
for topic in topics:
    print(topic)

lda_model = gensim.models.ldamodel.LdaModel(gensim_corpus, num_topics=8, id2word=gensim_dictionary, passes=15)
lda_model.save('gensim_model.gensim')
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)

lda_model = gensim.models.ldamodel.LdaModel(gensim_corpus, num_topics=4, id2word=gensim_dictionary, passes=20)
lda_model.save('gensim_model.gensim')
topics = lda_model.print_topics(num_words=10)
for topic in topics:
    print(topic)

test_doc = 'Great structures are build to remember an event happened in the history.'
test_doc = preprocess_text(test_doc)
bow_test_doc = gensim_dictionary.doc2bow(test_doc)

print(lda_model.get_document_topics(bow_test_doc))

print('\nPerplexity:', lda_model.log_perplexity(gensim_corpus))

'''
from gensim.models import CoherenceModel

coherence_score_lda = CoherenceModel(model=lda_model, texts=processed_data, dictionary=gensim_dictionary, coherence='c_v')
coherence_score = coherence_score_lda.get_coherence()

print('\nCoherence Score:', coherence_score)
'''

gensim_dictionary = gensim.corpora.Dictionary.load('gensim_dictionary.gensim')
gensim_corpus = pickle.load(open('gensim_corpus_corpus.pkl', 'rb'))
lda_model = gensim.models.ldamodel.LdaModel.load('gensim_model.gensim')

import pyLDAvis.gensim

lda_visualization = pyLDAvis.gensim.prepare(lda_model, gensim_corpus, gensim_dictionary, sort_topics=False)
pyLDAvis.display(lda_visualization)