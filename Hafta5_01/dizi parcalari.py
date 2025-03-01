import cv2, random

video = cv2.VideoCapture(0)
while video.isOpened():
    ret, yakalananResim = video.read()
    x,y,r = yakalananResim.shape
    cv2.imshow('Tüm resim',yakalananResim)
    maxx = random.randint(300,x)
    maxy = random.randint(300,x)
    parca = yakalananResim[maxx-300:maxx:-1,maxy-300:maxy]
    cv2.imshow('Parça',parca) 

    if cv2.waitKey(1)==ord('q'): break
video.release()
cv2.destroyAllWindows()