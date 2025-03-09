import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

ornek_metin = "Bugün hava çok güzel fakat bir şey canım istemiyor."

cumleler = sent_tokenize(ornek_metin)
kelimeler = word_tokenize(ornek_metin)

# print(cumleler)
# print("Kelime listesi: ",kelimeler)

# nltk.download("stopwords") # Bir kere indirilmesi gerek/yeterli
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopword_tr.words("turkish")) # stop word = anlamsız kelimeler
# stop_words_tr = set(stopwords.words("turkish")) # stop word = köksüz/anlamsız kelimeler
temizlenmis_liste = []
print("Stop words:",stop_words)
# print("Stop words Türkçe:",stop_words_tr)

for kelime in kelimeler:
   if kelime.casefold() not in stop_words: # casefold ile küçükharfe çevir.
        temizlenmis_liste.append(kelime)
       
print("Temizlenmiş liste: ",temizlenmis_liste)
