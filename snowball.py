from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer(language='english')

tokens = ['compute', 'computer', 'computed', 'computing']

for token in tokens:
    print(token + ' --> ' + stemmer.stem(token))