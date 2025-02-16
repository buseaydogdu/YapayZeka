# Resim açma, gri tonlu
import cv2
img = cv2.imread('images/squirrel11.jpg')
img2=cv2.imread('images/squirrel11.jpg',cv2.IMREAD_GRAYSCALE)

cv2.imshow('Deneme',img)
cv2.imshow('Deneme2',img2)

cv2.waitKey(0)# herhangi bir tuşa basılana kadar bekle
cv2.destroyAllWindows()
print(img2)
