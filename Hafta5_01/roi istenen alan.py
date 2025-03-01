import cv2, random, numpy

video = cv2.VideoCapture(0)
while video.isOpened():
    ret, yakalananResim = video.read()
    x,y,r = yakalananResim.shape
    # print(x,y)


    parca = yakalananResim[485:500, 500:560]
    yakalananResim[490:505, 530:590] = parca
    cv2.imshow('Tüm resim', yakalananResim)


    if cv2.waitKey(1)==ord('q'): break

video.release()
cv2.destroyAllWindows()    