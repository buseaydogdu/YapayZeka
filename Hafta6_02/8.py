import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def temizle(metin):
    metin = metin.lower()  # Küçük harfe çevir
    metin = re.sub(r'\d+', '', metin)  # Sayıları kaldır
    metin = metin.translate(str.maketrans('', '', string.punctuation))  # Noktalama kaldır
    kelimeler = metin.split()
    kelimeler = [kelime for kelime in kelimeler if kelime not in stopwords.words('turkish')]  # Stopwords kaldır
    lemmatizer = WordNetLemmatizer()
    kelimeler = [lemmatizer.lemmatize(k) for k in kelimeler]  # Lemmatization
    return " ".join(kelimeler)

veri = ["Bugün hava çok güzel.", "Ali, Efe ve Ece çay içecek.", "Selam söyle."]
temiz_veri = [temizle(cumle) for cumle in veri]
print(temiz_veri)


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(temiz_veri)
print(vectorizer.get_feature_names_out())  # Kelime listesi
print(X.toarray())  # Sayısal gösterim


