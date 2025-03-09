import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

tv = TextVectorization()
veri = [
    "Bugün hava çok güzel",
    "Ali, Efe ve Ece çay içecek",
    "Selam çok söyle",
    "Annem ve babam bugün bana geldi"
]

print(veri)

# adapt ile gerekli ön hazırlıklar yapılıyor.
tv.adapt(veri) # verileri eğitmek için. Verilerden sözlük oluşturarak her bir token(en küçük parçalar) indexlendi.
tv.get_vocabulary()

print("tv: ",tv.get_vocabulary())

vt = tv(veri) # vectorized text / sözlüğü vektörize etme

print("vt:",vt) # verinin sözlükteki indexlerini içeren sayısal şekli. 0 lar olmayan kelimelerin yerine..

