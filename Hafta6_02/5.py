import nltk
# nltk.download('punkt') # Bir kere yapılması gerek/yeter
from nltk.tokenize import sent_tokenize, word_tokenize

ornek_metin = "Bugün hava çok güzel ama canım bir şey istemiyor."

cumleler = sent_tokenize(ornek_metin)
kelimeler = word_tokenize(ornek_metin)

# print(cumleler)
# print("Kelime listesi: ",kelimeler)

# nltk.download("stopword_tr") # Bir kere indirilmesi gerek/yeterli
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import anlamsizk
anlamsizkelimeler=["çok"]
stop_words = set(anlamsizk.stopwords) # stop word = anlamsız kelimeler
# stop_words_tr = set(stopwords.words("turkish")) # stop word = köksüz/anlamsız kelimeler
temizlenmis_liste = []
# print("Stop words:",stop_words)
# print("Stop words Türkçe:",stop_words_tr)



for kelime in kelimeler:
   if kelime.casefold() not in stop_words: # casefold ile küçükharfe çevir.
        temizlenmis_liste.append(kelime)
       
print("Temizlenmiş liste: ",temizlenmis_liste)

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

kok_cikarilmis_liste = [stemmer.stem(kelime) for kelime in temizlenmis_liste]

print("Kökleri çıkarılmış liste:",kok_cikarilmis_liste) 
