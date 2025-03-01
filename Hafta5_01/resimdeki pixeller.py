# Temel resim işlemleri - itemset
import cv2, random

video = cv2.VideoCapture(0)
while video.isOpened():
    ret, yakalananResim = video.read()
    x,y,r = yakalananResim.shape

    for a in range(50,80):
        for b in range(50,80):
            # yakalananResim.itemset((a,b,0),0) # 1 Blue max=255
            # yakalananResim.itemset((a,b,1),0) # 2 Green max=255
            # yakalananResim.itemset((a,b,2),255) # 3 Red max=255
            yakalananResim[a,b,0]=0
            yakalananResim[a,b,1]=0
            yakalananResim[a,b,2]=255


    cv2.imshow('Tüm resim',yakalananResim)

    if cv2.waitKey(1)==ord('q'):break
cv2.release(0)
cv2.destroyAllWindows()



