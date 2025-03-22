# Çoklu veri(resim) resim gösterme
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
# import warnings; warnings.filterwarnings("ignore")
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() # x_train, y_train etiketleri

plt.figure(figsize=(4,5)) # 4x5=20 lik bir resim göstermek için
def resimleri_goster(adet):
    for n in range(adet):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(x_train[n], cmap='gray')
        plt.axis('off')
    plt.show()

resimleri_goster(17)

