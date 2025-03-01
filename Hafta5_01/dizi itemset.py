import cv2, random, numpy

video = cv2.VideoCapture(0)
while video.isOpened():
    ret, yakalananResim = video.read()
    x,y,r = yakalananResim.shape
    cv2.imshow('Tüm resim', yakalananResim)
    parca = cv2.resize(yakalananResim,(x//2,y//2))
    cv2.imshow('Şekli',parca)

    if cv2.waitKey(1)==ord('q'): break

video.release()
cv2.destroyAllWindows()    