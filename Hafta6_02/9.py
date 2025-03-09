# Örnek: Metin Sınıflandırma (Spam veya Normal Mesaj Ayrımı)
# pip install scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF + Naive Bayes Pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())  # Naive Bayes Modeli
])

# Örnek veri
veriler = ["Bu fırsatı kaçırmayın!", "Merhaba, nasılsın?", "Hemen tıkla ve kazan!"]
etiketler = [1, 0, 1]  # 1: Spam, 0: Normal

# Modeli eğit
text_clf.fit(veriler, etiketler)

# Tahmin yap
# yeni_metin = ["Bu harika kampanya için hemen katıl!"]
yeni_metin = ["Fırsatçı Ahmet"]
tahmin = text_clf.predict(yeni_metin)
print(tahmin)  # 1 (Spam) veya 0 (Normal) 
