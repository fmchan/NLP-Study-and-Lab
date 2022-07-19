import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from random import randint
import re

import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg as gut

print(gut.fileids())

macbeth_text = nltk.corpus.gutenberg.raw('shakespeare-macbeth.txt')
print(macbeth_text[:500])

def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence.lower()

macbeth_text = preprocess_text(macbeth_text)
print(macbeth_text[:500])

from nltk.tokenize import word_tokenize

macbeth_text_words = (word_tokenize(macbeth_text))
n_words = len(macbeth_text_words)
unique_words = len(set(macbeth_text_words))

print('Total Words: %d' % n_words)
print('Unique Words: %d' % unique_words)

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=3437)
tokenizer.fit_on_texts(macbeth_text_words)

vocab_size = len(tokenizer.word_index) + 1
word_2_index = tokenizer.word_index

print(macbeth_text_words[500])
print(word_2_index[macbeth_text_words[500]])

input_sequence = []
output_words = []
input_seq_length = 100

for i in range(0, n_words - input_seq_length , 1):
    in_seq = macbeth_text_words[i:i + input_seq_length]
    out_seq = macbeth_text_words[i + input_seq_length]
    input_sequence.append([word_2_index[word] for word in in_seq])
    output_words.append(word_2_index[out_seq])

print(input_sequence[0])

X = np.reshape(input_sequence, (len(input_sequence), input_seq_length, 1))
X = X / float(vocab_size)

y = to_categorical(output_words)

print("X shape:", X.shape)
print("y shape:", y.shape)

model = Sequential()
model.add(LSTM(800, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(LSTM(800, return_sequences=True))
model.add(LSTM(800))
model.add(Dense(y.shape[1], activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X, y, batch_size=64, epochs=10, verbose=1)

random_seq_index = np.random.randint(0, len(input_sequence)-1)
random_seq = input_sequence[random_seq_index]

index_2_word = dict(map(reversed, word_2_index.items()))

word_sequence = [index_2_word[value] for value in random_seq]

print(' '.join(word_sequence))

for i in range(100):
    int_sample = np.reshape(random_seq, (1, len(random_seq), 1))
    int_sample = int_sample / float(vocab_size)

    predicted_word_index = model.predict(int_sample, verbose=0)

    predicted_word_id = np.argmax(predicted_word_index)
    seq_in = [index_2_word[index] for index in random_seq]

    word_sequence.append(index_2_word[ predicted_word_id])

    random_seq.append(predicted_word_id)
    random_seq = random_seq[1:len(random_seq)]

final_output = ""
for word in word_sequence:
    final_output = final_output + " " + word

print(final_output)