# filtre
import cv2, random, numpy

video = cv2.VideoCapture(0)
while video.isOpened():
    ret, yakalananResim = video.read()
    x,y,r = yakalananResim.shape
    cv2.imshow("Resim",yakalananResim)

# img1 = cv2.imread('images/square1.jpg')

# img1[y1:y2, x1:x2,BGR indisi] = yenideger
yakalananResim[:,:][0] = [0,0,200] # Blue değerlerini 0 yap
# yakalananResim[:,:,1] = 0 # Green değerlerini 0 yap
# img1[:,:,2] = 0 # Red değerlerini 0 yap

video.release()
cv2.destroyAllWindows()

