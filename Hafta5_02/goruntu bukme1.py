import cv2, random, numpy 

video = cv2.VideoCapture(0)
while video.isOpened():
    ret, goruntu = video.read()
    cv2.imshow("Orjinal Goruntu", goruntu)

    sutun,satir =goruntu.shape[:2]

    donmeSekli = cv2.getRotationMatrix2D((satir//4,sutun//4),30,1)
    donmusSekli = cv2.warpAffine(goruntu, donmeSekli, (satir,sutun))

    cv2.imshow("Cevirilmis sekli", donmusSekli)

    if cv2.waitKey(1)==ord("q"): break

video.release()
cv2.destroyAllWindows() 
