from gensim.models import KeyedVectors
filename = 'D:\downloads\GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)

# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'son'], negative=['man'], topn=1)
print(result)