from PyQt6.QtWidgets import *


class Pencere(QMainWindow):
    def tahmin(self,ted): # ted : tahmin edilecek değer
        import pandas as pd, numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error

        # Örnek veri seti
        data = {'TV Reklamı': [230.1, 44.5, 17.2, 151.5, 180.8],
                'Satış': [22.1, 10.4, 9.3, 18.5, 12.9]}
        df = pd.DataFrame(data)

        # Bağımsız ve bağımlı değişkenler
        X = df[['TV Reklamı']]
        y = df['Satış']

        # Veriyi eğitim ve test setlerine ayırma
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modeli oluşturma ve eğitme
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Test verisi üzerinde tahmin yapma
        y_pred = model.predict(X_test)

        # Modelin performansını değerlendirme
        mse = mean_squared_error(y_test, y_pred)
        print(f'MSE=Mean Squared Error/Hata Karelerinin Ortalaması: {mse}')

        # Model katsayıları
        slope = model.coef_[0]  # TV Reklamı'nın katsayısı
        intercept = model.intercept_  # Y-intercept (modelin kesişim noktası)

        print(f"Model Katsayıları: Slope = {slope}, Intercept = {intercept}")

        # Kullanıcıdan reklam bütçesi girmesini isteyelim
        # reklam_butcesi = float(input("Reklam bütçesini girin (birim): "))
        reklam_butcesi = float(ted)

        # Satış tahmini için DataFrame kullanımı (Feature name uyumluluğunu korur)
        satış_tahmini = model.predict(pd.DataFrame({'TV Reklamı': [reklam_butcesi]}))

        print(f"{reklam_butcesi} birimlik reklam harcaması için tahmin edilen satış: {satış_tahmini[0]:.2f} birim")
        return satış_tahmini

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reklam geliri tahmini")

        icerik = QVBoxLayout()
          #icerik = QHBoxLayout()
        icerik.addWidget(QLabel("Reklam bütçenizi girin: "))
        self.veriKutusu = QLineEdit()
        icerik.addWidget(self.veriKutusu)
        button = QPushButton("Tahmin Et")
        icerik.addWidget(button)
        button.clicked.connect(self.mesaj)

        self.sonuc_label = QLabel("Tahmini satış artışı: ")
        icerik.addWidget(self.sonuc_label)
        araclar = QWidget()
        araclar.setLayout(icerik)
        self.setCentralWidget(araclar)

    def mesaj(self):
        # self.veriKutusu.setText("Tıklandı")
        # print("Tıklandı")
        # cmdeger = int(self.veriKutusu.text()) * 2
        # self.veriKutusu.setText(str(cmdeger))
        reklam_butcesi = self.veriKutusu.text()
        tahmin_sonucu = self.tahmin(reklam_butcesi)

        # self.veriKutusu.setText(str(tahmin_sonucu))
        mevcut = self.sonuc_label.text()
        self.sonuc_label.setText(mevcut+str(tahmin_sonucu))


uygulama = QApplication([])

pencere = Pencere()
pencere.show()

uygulama.exec()