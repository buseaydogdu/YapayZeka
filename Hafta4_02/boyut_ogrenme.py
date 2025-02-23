import cv2, numpy as np

video = cv2.VideoCapture(0)

while video.isOpened():
    aa,alinanGoruntu = video.read()
    yeniDizi = np.hsplit(alinanGoruntu,4)
    # print(f'Yükseklik: {yukseklik}, Genişlik:{genislik}, renk:{renk}')

    yukseklik, genislik, renk = yeniDizi[1].shape

    yeniResim= np.full([yukseklik,genislik,3], [250,250,250], dtype=np.uint8)
    gosterilecekSekli = np.concatenate((yeniDizi[0],yeniResim,yeniDizi[2]),axis=1)
    # yenisi = np.array_split(alinanGoruntu,2) 
    # dparca1 = np.hsplit(yenisi[0],2)
    # dparca2 = np.hsplit(yenisi[1],2)
 

    cv2.imshow("Alınan görüntü1:", gosterilecekSekli)
    tus = cv2.waitKey(1)
    if tus == 97 or tus == ('a'): break


video.release()
cv2.destroyAllWindows()
# cv2.imshow("Alınan görüntü2:",dparca1[1])
    # cv2.imshow("Alınan görüntü3:",dparca1[0])
    # cv2.imshow("Alınan görüntü4:",dparca1[1])
    # birlesik = np.concatenate((dparca1[0],dparca2[1],dparca2[0]), axis=1)
    # cv2.imshow("Alınan görüntü1:",birlesik)
    
