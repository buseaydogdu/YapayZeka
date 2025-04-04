import cv2

cam = cv2.VideoCapture(0)
cam.get(5) # genişlik (width)
cam.set(5,150)
while True:
    ret, alinanOrj = cam.read()
    alinanSB = cv2.cvtColor(alinanOrj, cv2.COLOR_BGR2GRAY)

    if not cam.isOpened() or not ret:
        print("Kameradan görüntü alınamıyor.")
        exit()

    cv2.imshow("Orjinal Goruntu",alinanOrj)

    cv2.imshow("Siyah Beyaz Goruntu",alinanSB)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        print("Görüntü kapatıldı"); break

cam.release()
cv2.destroyAllWindows() 
