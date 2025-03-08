# NLTK ile tokenize (parçalama)
# pip install nltk

# import nltk
# nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize, word_tokenize
metin = "Bu gün hava çok güzel. Dışarı çıkıp biraz dolaştım. Arkadaşlarla bir yerde oturduk."
cumleler = sent_tokenize(metin)
print (cumleler)

kelimeler = word_tokenize(metin)
print(kelimeler)

