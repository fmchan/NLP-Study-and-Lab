import spacy
sp = spacy.load('zh_core_web_sm')

from spacy import displacy

sen = sp(u"我想屌你老母! 可以嗎? I like to play football. I hated it in my childhood though")
displacy.render(sen, style='dep', jupyter=True, options={'distance': 85})
displacy.serve(sen, style='dep', options={'distance': 120})