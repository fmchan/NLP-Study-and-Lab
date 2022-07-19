import spacy

sp = spacy.load('zh_core_web_sm')
sentence = sp("屌你老母! 你好")
for word in sentence:
    print(word.text,  word.pos_, word.dep_)
for entity in sentence.ents:
    print(entity.text + ' - ' + entity.label_ + ' - ' + str(spacy.explain(entity.label_)))

sp = spacy.load('en_core_web_sm')

sentence2 = sp(u"Manchester United isn't looking to sign any forward.")
for word in sentence2:
    print(word.text,  word.pos_, word.dep_)

sentence5 = sp(u'Manchester United is looking to sign Harry Kane for $90 million')
for entity in sentence5.ents:
    print(entity.text + ' - ' + entity.label_ + ' - ' + str(spacy.explain(entity.label_)))

from spacy.matcher import Matcher
m_tool = Matcher(sp.vocab)

p1 = [{'LOWER': 'quickbrownfox'}]
p2 = [{'LOWER': 'quick'}, {'IS_PUNCT': True}, {'LOWER': 'brown'}, {'IS_PUNCT': True}, {'LOWER': 'fox'}]
p3 = [{'LOWER': 'quick'}, {'LOWER': 'brown'}, {'LOWER': 'fox'}]
p4 =  [{'LOWER': 'quick'}, {'LOWER': 'brownfox'}]
m_tool.add('QBF', [p1, p2, p3, p4])

sentence = sp(u'The quick-brown-fox jumps over the lazy dog. The quick brown fox eats well. \
               the quickbrownfox is dead. the dog misses the quick brownfox')
phrase_matches = m_tool(sentence)
print(phrase_matches )

for match_id, start, end in phrase_matches:
    string_id = sp.vocab.strings[match_id]  
    span = sentence[start:end]                   
    print(match_id, string_id, start, end, span.text)
