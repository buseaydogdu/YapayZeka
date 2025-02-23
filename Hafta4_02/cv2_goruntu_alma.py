import cv2, numpy as np

video = cv2.VideoCapture(0)

while video.isOpened():
    aa,bb = video.read()
    print(bb)
    yenisi = np.array_split(bb,4)   
    # cv2.imshow("Alınan görüntü",bb)
    cv2.imshow("Alınan görüntü1",yenisi[0])
    cv2.imshow("Alınan görüntü2",yenisi[1])
    cv2.imshow("Alınan görüntü3",yenisi[2])
    cv2.imshow("Alınan görüntü4",yenisi[3])
    # dparca1 = np.hsplit(yenisi[0],2)
    # dparca1 = np.hsplit(yenisi[1],2)

    # cv2.imshow("Alınan görüntü1:",dparca1[0])
    # cv2.imshow("Alınan görüntü2:",dparca1[1])
    # cv2.imshow("Alınan görüntü3:",dparca1[0])
    # cv2.imshow("Alınan görüntü4:",dparca1[1])


    tus = cv2.waitKey(1)
    if tus == 97 or tus == ('a'): break



video.release()
cv2.destroyAllWindows()

