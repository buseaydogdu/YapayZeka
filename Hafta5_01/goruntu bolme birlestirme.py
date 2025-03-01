#  görüntünün yarısı renkli, yarısı siyah beyaz olarak göstersin.
import cv2, numpy as np

video = cv2.VideoCapture(0)

while video.isOpened():
    aa,alinanGoruntu = video.read()
    # print(alinanGoruntu)
    yenisi = np.array_split(alinanGoruntu,2) 
    dparca1 = np.hsplit(yenisi[0],2)
    dparca2 = np.hsplit(yenisi[1],2)
    dparca2 = cv2.cvyColor(dparca2[1],cv2.COLOR_BGR2GRAY)
    x, y = dparca2[1].shape
    print(x,y)
    # for a in range(x):
    #     for b in range(y):
    #         print(dparca2[1],[x],[y])



    # cv2.imshow("Alınan görüntü1:",dparca1[0])
    # cv2.imshow("Alınan görüntü2:",dparca1[1])
    # cv2.imshow("Alınan görüntü3:",dparca1[0])
    # cv2.imshow("Alınan görüntü4:",dparca1[1])
    # birlesik = np.concatenate((dparca1[0],dparca2[1],dparca2[0]), axis=1)
    # cv2.imshow("Alınan görüntü1:",birlesik)
    tus = cv2.waitKey(1)
    if tus == 97 or tus == ('a'): break


video.release()
cv2.destroyAllWindows()

