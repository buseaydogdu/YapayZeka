import cv2, numpy as np

# Yüz tespiti için Haar Cascade Classifier'ı yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0) # Kamerayı aç

while True:
    ret, frame = cap.read() # Kameradan bir kare al
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Görüntüyü gri tonlamaya çevir

    # Yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) # Yüzün üzerine dikdörtgen çiz

        # Yüzün alt kısmından burnu tahmin et (burun altı kısmı genellikle yüzün alt bölgesindedir)
        nose_region = frame[y + h//3:y + 2*h//3, x + w//4:x + 3*w//4]  # Burun altı kısmını seçiyoruz

        # Kafanın üst kısmına burnu taşıyalım (baş kısmı genellikle yüzün üst kısmında yer alır)
        frame[y - h//3:y - h//3 + nose_region.shape[0], x + w//4:x + 3*w//4] = nose_region

    cv2.imshow('Kameradan Yüz Tespiti ve Burun Taşıma', frame) # Sonuç görüntüsünü göster

    if cv2.waitKey(1) & 0xFF == ord('q'): break # 'q' tuşuna basarak çıkılabilir

cap.release()# Kamerayı kapat
cv2.destroyAllWindows()
