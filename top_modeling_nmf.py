import pandas as pd
import numpy as np
from IPython.display import display

reviews_datasets = pd.read_csv(r'D:\downloads\Reviews.csv')
reviews_datasets = reviews_datasets.head(20000)
reviews_datasets.dropna()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
doc_term_matrix = tfidf_vect.fit_transform(reviews_datasets['Text'].values.astype('U'))

from sklearn.decomposition import NMF

nmf = NMF(n_components=5, random_state=42)
nmf.fit(doc_term_matrix )

import random

for i in range(10):
    random_id = random.randint(0,len(tfidf_vect.get_feature_names_out()))
    print(tfidf_vect.get_feature_names_out()[random_id])

first_topic = nmf.components_[0]
top_topic_words = first_topic.argsort()[-10:]

for i in top_topic_words:
    print(tfidf_vect.get_feature_names_out()[i])

for i,topic in enumerate(nmf.components_):
    print(f'Top 10 words for topic #{i}:')
    print([tfidf_vect.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
    print('\n')

topic_values = nmf.transform(doc_term_matrix)
reviews_datasets['Topic'] = topic_values.argmax(axis=1)
reviews_datasets.head()
display(reviews_datasets)