import bs4 as bs  
import urllib.request  
import re  
import nltk

scrapped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Artificial_intelligence')  
article = scrapped_data .read()

parsed_article = bs.BeautifulSoup(article,'lxml')

paragraphs = parsed_article.find_all('p')

article_text = ""

for p in paragraphs:  
    article_text += p.text
    
    
processed_article = article_text.lower()  
processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )  
processed_article = re.sub(r'\s+', ' ', processed_article)

# Preparing the dataset
all_sentences = nltk.sent_tokenize(processed_article)

all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

# Removing Stop Words
from nltk.corpus import stopwords
for i in range(len(all_words)):
    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]

from gensim.models import Word2Vec

word2vec = Word2Vec(all_words, min_count=2)
vocabulary = word2vec.wv.index_to_key
print(len(vocabulary))
v1 = word2vec.wv['artificial']
sim_words = word2vec.wv.most_similar('intelligence')
print(sim_words)

import spacy
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import PhraseMatcher
phrase_matcher = PhraseMatcher(nlp.vocab)
phrases = ['machine learning', 'robots', 'intelligent agents']

patterns = [nlp(text) for text in phrases]
phrase_matcher.add('AI', None, *patterns)
sentence = nlp (processed_article)

matched_phrases = phrase_matcher(sentence)

for match_id, start, end in matched_phrases:
    string_id = nlp.vocab.strings[match_id]  
    span = sentence[start:end]                   
    print(match_id, string_id, start, end, span.text)