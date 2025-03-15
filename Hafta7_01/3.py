import cv2
import numpy as np
from sklearn.svm import SVC  # Basit bir SVM modeli
from sklearn.preprocessing import StandardScaler

# 1. OpenCV ile Kamerayı Aç
cap = cv2.VideoCapture(0)  # 0, varsayılan kamerayı kullanır
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 2. Örnek eğitim verisi (Gerçek modelde buraya eğitim süreci eklenebilir)
X_train = np.array([[100,100],[200, 200],[300,300]])  # Örnek yüz boyutları
y_train = np.array([0, 1, 2])  # Örnek etiketler (0: Normal, 1: Yakın gibi)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = SVC(kernel="linear")  # Basit bir SVM modeli
model.fit(X_train, y_train)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # 3. Yüzü Çerçeve İçinde Göster
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 4. Yüzü Makine Öğrenmesi Modeline Gönder
        face_features = np.array([[w, h]])  # Basit olarak yüz genişliği ve yüksekliği
        face_features = scaler.transform(face_features)
        prediction = model.predict(face_features)

        # label = f"Cok yakinsin w:{w}, h:{h}" if prediction[0] == 0 else f"Cok yakinsin w:{w}, h:{h}"
        if prediction[0] == 0: label = f"Cok uzaksin w:{w}, h:{h}"
        if prediction[0] == 1: label = f"Normal w:{w}, h:{h}"
        if prediction[0] == 2: label = f"Cok yakinsin w:{w}, h:{h}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Kamera Görüntüsü", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # 'q' ile çıkış
        break

cap.release()
cv2.destroyAllWindows()
