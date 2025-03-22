# DOĞRULUK KONTROLÜ
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
# import warnings; warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical, plot_model

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() # x_train : 28x28 eğitim görselleri, y_train görüntü etiketleri
print(f"Eğitim setinin boyutu:\t {x_train.shape}x{y_train.shape}")
print(f"Test setinin boyutu:\t {x_test.shape}x{y_test.shape}")

num_labels = len (np.unique(y_train))
print("Farklı etiket sayısı: ", num_labels)

def resim_uzerinde_deger_göster(resim):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.imshow(resim, cmap='gray')
   
    genislik, yukseklik = resim.shape  
    esik = resim.max() / 2.5  
    for x in range(genislik):
        for y in range(yukseklik):
            ax.annotate(
                str(round(resim[x, y], 2)),
                xy=(y, x),
                color="blue" if y % 2 == 0 else 'red',
                ha="center", va="center", fontsize=7  # # Ortalamak ve küçült
            )    
    plt.show()

def resim_uzerinde_deger_gösterSB(resim):
    # Eğer resim 3D (28, 28, 1) formatında ise, sadece 2D kısmı alıyoruz.
    if len(resim.shape) == 3:
        resim = resim.squeeze()  # 3D'yi 2D'ye indirger
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.imshow(resim, cmap='gray')
   
    genislik, yukseklik = resim.shape  # Şimdi 2D olduğu için doğru şekilde unpack edebiliriz.
    esik = resim.max() / 2.5
    for x in range(genislik):
        for y in range(yukseklik):
            ax.annotate(
                str(round(resim[x, y], 2)),
                xy=(y, x),
                color="blue" if y % 2 == 0 else 'red',
                ha="center", va="center", fontsize=7
            )    
    plt.show()

# resim_uzerinde_deger_göster(x_train[2])
resim_uzerinde_deger_gösterSB(x_train[2])

# VERİYİ HAZIRLAMA İÇİN 3 AŞAMA TAKİP EDİLİR
# 1- Encoding / veriyi sayısallaştırma
# 2- Reshaping / veriyi yeniden şekillendirme
# 3- Normalization / Standardizasyon / Aykırılıkları yok etme

# 1- Encoding / veriyi sayısallaştırma
# Bağımlı değişken (label, output, hedef değişken) kategorik olduğu için,
# sayısal formata çevirmek amacıyla "one-hot encoding" yapıyoruz.
# Örneğin, rakamlar 0-9 arasında olduğu için, her rakam 10 elemanlı bir vektör olarak gösterilecek.

print("\n\ny_train[0:5] : ", y_train[0:5])  # İlk 5 eğitim verisinin etiketlerini yazdır.
print("\n\nto_categorical(y_train[0:5]) : \n", to_categorical(y_train[0:5]))  # İlk 5 eğitim etiketini one-hot encoding ile dönüştür.
y_train = to_categorical(y_train)
print("\n\nto_categorical(y_test[0:5]) : \n", to_categorical(y_test[0:5]))  # İlk 5 test etiketini de dönüştür.
y_test = to_categorical(y_test)

# 2- Reshaping / veriyi yeniden şekillendirme
# MNIST veri kümesinde görüntüler 28x28 boyutunda ve tek kanallı (grayscale) olduğu için,
# veriyi (num_samples, height, width, channels) formatına getirmeliyiz.
# Keras'ın Convolutional Neural Network (CNN) modelleri bu formatı bekler.

print("\n\nx_train.shape[1] : ", x_train.shape[1])  # Görsellerin boyutunu (yükseklik) ekrana yazdır.
image_size = x_train.shape[1]
print("\n\nx_train.shape1 : ", x_train.shape)  # Eğitim veri kümesinin mevcut boyutunu yazdır.
print("\n\nx_test.shape1 : ", x_test.shape)  # Test veri kümesinin mevcut boyutunu yazdır.

# (num_samples, 28, 28, 1) olacak şekilde yeniden şekillendiriyoruz.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

print("\n\nx_train.shape2 : ", x_train.shape)  # Yeniden şekillendirilmiş eğitim veri kümesinin boyutu.
print("\n\nx_test.shape2 : ", x_test.shape)  # Yeniden şekillendirilmiş test veri kümesinin boyutu.

# 3- Normalization / Standardizasyon / Aykırılıkları yok etme
# Görsellerdeki piksel değerleri 0 ile 255 arasında değişir.
# Modelin daha iyi öğrenmesi için tüm değerleri 0 ile 1 arasına ölçeklendirmemiz gerekir.
# Bunu yapmak için tüm değerleri 255'e bölüyoruz.

x_train = x_train.astype('float32') / 255  # Eğitim veri kümesini float32 formatına çevirip 255'e bölerek normalleştiriyoruz.
x_test = x_test.astype('float32') / 255  # Test veri kümesini aynı şekilde normalleştiriyoruz.

# Normalizasyon sonrası görselleri incelemek için, piksel değerlerini ekrana yazdırabiliriz.
# resim_uzerinde_deger_gösterSB(x_train[2])  # Normalleştirilmiş verinin görselleştirilmesi (isteğe bağlı).

print("\n\nx_train[2] : \n", x_train[2])  # Normalleştirilmiş bir örnek görselin yeni piksel değerlerini ekrana yazdır.

# MODEL KURMA
model = tf.keras.Sequential([
    Flatten(input_shape=(28,28,1)),
    Dense(units=128, activation='relu', name='layer1'), # unit = 128 unit: nöron sayısı ,28x28 resimleri tanımlayabilecek feature/özellik/nöron sayısı, activation='relu' insan beynindeki gibi gereksiz bazı nöronları sönümlendiren bazılarını güçlendiren bir fonksiyon. gizli katmanlarda relu kullanılır.
    # relu duruma göre bir nöronu sönümleyen(sıfır yapan) yada ateşleyen bir fonksindur.
    Dense(units=num_labels, activation='softmax', name='output_layer') # activation='softmax' : çok sınıflı bir sınıflandırma problemi için softmax, iki sınıflı bir sınıflandırma problemi için sigmoid kullanılır.    
])

model.compile(loss='categorical_crossentropy', # lose: hata/kayıp hesaplama yöntemi/hata değerlendirme metriği. Çok sınıflı olduğu için categorical_crossentropy
              optimizer='adam', # loss fonksiyonunu optimize edecek algoritma. adam, Stochastic Gradient Descent,
              metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 'accuracy' ]
              # metrics olarak Precision, Recall ve accuracy özellikle dengesiz sınıflarda Precision ve Recall değerleri daha anlamlı olabilir.
              )

model.summary()

model.fit(x_train, y_train,epochs=5, batch_size=128, validation_data=(x_test, y_test)) # epoch : tur sayısı, batch_size: gradian hesaplarında ağırlık güncelleme küme sayısı. 128 = her iterasyonda 128 gözlem birimi dikkate alınıp işlemler yapıldıktan sonra sonraki iterasyona geç.

history = model.fit(x_train, y_train,epochs=5, batch_size=128, validation_data=(x_test, y_test)) # epoch : tur sayısı, batch_size: gradian hesaplarında ağırlık güncelleme küme sayısı. 128 = her iterasyonda 128 gözlem birimi dikkate alınıp işlemler yapıldıktan sonra sonraki iterasyona geç.

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5))

# Grafik 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], color='b', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], color='r', label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.ylim([min(plt.ylim()), 1])
plt.title('Eğitim ve Test Başarım Grafiği', fontsize=16)

# Grafik 2: Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], color='b', label='Training Loss')
plt.plot(history.history['val_loss'], color='r', label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.ylim([0, max(plt.ylim())])
plt.title('Eğitim ve Test Kayıp Grafiği', fontsize=16)

plt.show()

loss, precision, recall, acc = model.evaluate(x_test, y_test, verbose=False)

print("\nTest Accuracy(Test Doğruluğu): %.1f%%" % (100.0 * acc))
print("\nTest Loss(Test kayıp/hatası): %.1f%%" % (100.0 * loss))
print("\nTest Precision (Test Kesinliği/Duyarlılığı): %.1f%%" % (100.0 * precision))
print("\nTest Recall (Test Geri Çağırma/Hatırlama): %.1f%%" % (100.0 * recall))


print("\n\nx_train.shape1 : ", x_train.shape)  # Eğitim veri kümesinin mevcut boyutunu yazdır.
print("\n\nx_test.shape1 : ", x_test.shape)  # Test veri kümesinin mevcut boyutunu yazdır.

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5))

# Grafik 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], color='b', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], color='r', label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.ylim([min(plt.ylim()), 1])
plt.title('Eğitim ve Test Başarım Grafiği', fontsize=16)

# Grafik 2: Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], color='b', label='Training Loss')
plt.plot(history.history['val_loss'], color='r', label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.ylim([0, max(plt.ylim())])
plt.title('Eğitim ve Test Kayıp Grafiği', fontsize=16)

plt.show()

from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


