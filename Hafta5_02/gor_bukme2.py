import cv2, random, numpy as np

video = cv2.VideoCapture(0)
while video.isOpened():
    ret, goruntu = video.read()
    cv2.imshow("Orjinal Goruntu", goruntu)

    # sutun,satir =goruntu.shape[:2]
    sutun,satir, renk =goruntu.shape

    baslangic = np.float32([[0,0],[sutun,0],[0,satir]])
    yeniyer   = np.float32([[50,50],[sutun//2,100],[200,satir//2]])

    bukmeSekli = cv2.getAffineTransform(baslangic,yeniyer)
    bukulmusSekli = cv2.warpAffine(goruntu, bukmeSekli, (satir,sutun))

    # cv2.imshow("Bukulmus sekli", bukulmusSekli)


    # donmeSekli = cv2.getRotationMatrix2D((satir//4,sutun//4),30,1)
    # donmeSekli = cv2.warpAffine(goruntu, donmeSekli, (satir,sutun))

    cv2.imshow("Cevirilmis sekli", bukulmusSekli)

    if cv2.waitKey(1)==ord("q"): break

video.release()
cv2.destroyAllWindows() 
