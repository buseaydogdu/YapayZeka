# traspose işlemi
import cv2, random, numpy 

video = cv2.VideoCapture(0)
while video.isOpened():
    ret, yakalananResim = video.read()

    cv2.imshow('Tüm resim', yakalananResim)
    

    cv2.imshow('Tüm resim2', cv2.transpose(yakalananResim))
    
    if cv2.waitKey(1)==ord('q'): break

video.release()
cv2.destroyAllWindows() 
