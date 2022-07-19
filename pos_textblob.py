from textblob import TextBlob
document = ("In computer science, artificial intelligence (AI), \
            sometimes called machine intelligence, is intelligence \
            demonstrated by machines, in contrast to the natural intelligence \
            displayed by humans and animals. Computer science defines AI \
            research as the study of \"intelligent agents\": any device that \
            perceives its environment and takes actions that maximize its\
            chance of successfully achieving its goals.[1] Colloquially,\
            the term \"artificial intelligence\" is used to describe machines\
            that mimic \"cognitive\" functions that humans associate with other\
            human minds, such as \"learning\" and \"problem solving\".[2]")
text_blob_object = TextBlob(document)
document_sentence = text_blob_object.sentences

print(document_sentence)
print(len(document_sentence))

document_words = text_blob_object.words

print(document_words)
print(len(document_words))

from textblob import Word

word1 = Word("apples")
print("apples:", word1.lemmatize())

word2 = Word("media")
print("media:", word2.lemmatize())

word3 = Word("greater")
print("greater:", word3.lemmatize("a"))

for word, pos in text_blob_object.tags:
    print(word + " => " + pos)

text = ("Football is a good game. It has many health benefit")
text_blob_object = TextBlob(text)
print(text_blob_object.words.pluralize())

text = ("Footballs is a goods games. Its has many healths benefits")

text_blob_object = TextBlob(text)
print(text_blob_object.words.singularize())

text_blob_object = TextBlob(document)
for noun_phrase in text_blob_object.noun_phrases:
    print(noun_phrase)

text_blob_object = TextBlob(document)
print(text_blob_object.word_counts['intelligence'])
print(text_blob_object.words.count('intelligence'))
print(text_blob_object.words.count('intelligence', case_sensitive=True))
text_blob_object = TextBlob(document)
print(text_blob_object.noun_phrases.count('artificial intelligence'))

text = "I love to watch football, but I have never played it"
text_blob_object = TextBlob(text)

print(text_blob_object.upper())

text = "I LOVE TO WATCH FOOTBALL, BUT I HAVE NEVER PLAYED IT"
text_blob_object = TextBlob(text)

print(text_blob_object.lower())

text = "I love to watch football, but I have never played it"
text_blob_object = TextBlob(text)
for ngram in text_blob_object.ngrams(2):
    print(ngram)

text = "I love to watchf footbal, but I have neter played it"
text_blob_object = TextBlob(text)

print(text_blob_object.correct())

text_blob_object_french = TextBlob(u'Salut comment allez-vous?')
#print(text_blob_object_french.translate(to='en'))

text_blob_object_arabic = TextBlob(u'مرحبا كيف حالك؟')
#print(text_blob_object_arabic.translate(to='en'))

text_blob_object = TextBlob(u'Hola como estas?')
#print(text_blob_object.detect_language())

train_data = [
    ('This is an excellent movie', 'pos'),
    ('The move was fantastic I like it', 'pos'),
    ('You should watch it, it is brilliant', 'pos'),
    ('Exceptionally good', 'pos'),
    ("Wonderfully directed and executed. I like it", 'pos'),
    ('It was very boring', 'neg'),
    ('I did not like the movie', 'neg'),
    ("The movie was horrible", 'neg'),
    ('I will not recommend', 'neg'),
    ('The acting is pathetic', 'neg')
]
test_data = [
    ('Its a fantastic series', 'pos'),
    ('Never watched such a brillent movie', 'pos'),
    ("horrible acting", 'neg'),
    ("It is a Wonderful movie", 'pos'),
    ('waste of money', 'neg'),
    ("pathetic picture", 'neg')
]

from textblob.classifiers import NaiveBayesClassifier
classifier = NaiveBayesClassifier(train_data)
print(classifier.classify("It is very boring"))
print(classifier.classify("It's a fantastic series"))

sentence = TextBlob("It's a fantastic series.", classifier=classifier)
print(sentence.classify())
print(classifier.accuracy(test_data))

classifier.show_informative_features(3)