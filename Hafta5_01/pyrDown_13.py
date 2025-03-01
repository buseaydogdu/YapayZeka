# pyrDown
import cv2, random,  numpy

video = cv2.VideoCapture(0)
while video.isOpened():
    ret, yakalananResim = video.read()

    cv2.imshow('Tüm resim', yakalananResim)
    kr =cv2.pyrDown(yakalananResim)
    cv2.imshow('Tüm resim1',kr)
    cv2.imshow('Tüm resim1', cv2.pyrDown(yakalananResim))
    
    if cv2.waitKey(1)==ord('q'): break

cv2.waitKey(0)
cv2.destroyAllWindows()

