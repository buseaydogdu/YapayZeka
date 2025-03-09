import nltk
# nltk.download('punkt') # Bir kere yapılması gerek/yeter
from nltk.tokenize import sent_tokenize, word_tokenize

ornek_metin = """yandaki ornek_metin i kullanabilirsiniz."""

cumleler = sent_tokenize(ornek_metin)
kelimeler = word_tokenize(ornek_metin)

# print(cumleler)
# print("Kelime listesi: ",kelimeler)

# nltk.download("stopwords") # Bir kere indirilmesi gerek/yeterli
from nltk.corpus import stopwords

stop_words_tr = set(stopwords.words("turkish")) # stop word = köksüz/anlamsız kelimeler
temizlenmis_liste = []
print("Stop words Türkçe:",stop_words_tr)

for kelime in kelimeler:
   if kelime.casefold() not in stop_words_tr: # casefold ile küçükharfe çevir.
        temizlenmis_liste.append(kelime)
       
print("Temizlenmiş liste: ",temizlenmis_liste)

# from nltk.stem import PorterStemmer
# stemmer = PorterStemmer()
# kok_cikarilmis_liste = [stemmer.stem(kelime) for kelime in temizlenmis_liste]

# pip install snowballstemmer
from snowballstemmer import TurkishStemmer
turkStem=TurkishStemmer()
# turkStem.stemWord("ekmekler") #ekmek

kok_cikarilmis_liste = [turkStem.stemWord(kelime) for kelime in temizlenmis_liste]

print("Kökleri çıkarılmış liste:",kok_cikarilmis_liste)
