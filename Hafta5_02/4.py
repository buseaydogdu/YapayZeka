# perspective trasformation/bükme
import cv2, numpy as np

goruntu = cv2.imread('images/squirrel11.jpg')
cv2.imshow("Orjinal Goruntu", goruntu)

sutun, satir = goruntu.shape[:2]

# float32([[üstsol x,y],[üstsağ x,y],[altsol x,y],[altsağ x,y]])
baslangic = np.float32([[0,0],[sutun,0],[0,satir],[sutun,satir]])
yeniyer   = np.float32([[90,60],[sutun*.7,50],[80,satir//2],[300,400]])

bukmeSekli = cv2.getPerspectiveTransform(baslangic,yeniyer)
bukulmusSekli = cv2.warpPerspective(goruntu, bukmeSekli, (satir,sutun))

cv2.imshow("Bukulmus sekli", bukulmusSekli)

cv2.waitKey(0)
cv2.destroyAllWindows()



