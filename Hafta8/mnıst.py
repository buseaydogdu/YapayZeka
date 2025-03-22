# MNIST veri setini indirip görüntüleme
# pip install tensorflow numpy matplotlib   # kütüphaneler yükle
import tensorflow as tf, matplotlib.pyplot as plt, numpy as np

# 28x28 piksellik El yazısı rakamlar içeren MNIST veri setini yükle (yaklışık 11MB)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# komutunu ilk kez çalıştırdığında, MNIST veri seti bilgisayar yoksa Google’ın sunucularından indirilir ve
# .keras/datasets/ klasörüne mnist.npz olarak kaydeder. (C:\Users\erdincdonmez\.keras\datasets gibi)
# Daha sonraki çalışmalarda bu dosya yerindeyse ve bir arıza yoksa tekrar indirmez, yerel dosyadan yükler..
# Windows: C:\Users\KullanıcıAdın\.keras\datasets\mnist.npz
# Linux/macOS: /home/kullanici/.keras/datasets/mnist.npz

# Eğitim ve test setlerinin boyutlarını yazdır
print(f"Eğitim setinin boyutu:\t {x_train.shape}x{y_train.shape}")
print(f"Test setinin boyutu:\t {x_test.shape}x{y_test.shape}")

num_labels = len(np.unique(y_train)) # Veri setinde kaç farklı etiket (rakam) olduğunu belirle
print("Farklı etiket sayısı:", num_labels)  # Çıkış: 10 (0'dan 9'a kadar rakamlar)

print("\n\nx_train[2] verisi: ", x_train[2])
print("\n\nx_train[2][14,10] verisi: ", x_train[2][14,10])
print("\n\nx_train[2].sum() : ", x_train[2].sum())
print("\n\nx_train[2][14:20, 10:20] : \n", x_train[2][14:20, 10:20])
print("\n\nx_train[2][14:20, 10:20].mean() : ", x_train[2][14:20, 10:20].mean())

