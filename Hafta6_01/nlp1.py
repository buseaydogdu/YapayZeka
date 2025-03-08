#  pip install spacy
import spacy
nlp = spacy.load(tr_core_news_sm)
metin = nlp("merhaba, bugün hava çok güzel")

for cumle in metin.sents:
    print(cumle)

for kelime in cumle:
    print(kelime)
