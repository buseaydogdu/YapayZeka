# Resim kaydetme
import cv2
img = cv2.imread('images/squirrel11.jpg')
img2=cv2.imread('images/squirrel11.jpg',cv2.IMREAD_GRAYSCALE)

cv2.imshow("Deneme",img2)
tus = cv2.waitKey(3000)
# cv2.imwrite('images/squirrel11.jpg',img)
cv2.destroyAllWindows()

