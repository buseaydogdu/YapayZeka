#  pip install opencv-python
# https://www.tamindir.com/indir/paintnet/
# Resim açma
import cv2
img = cv2.imread('squirrel11.jpg')
print(img)
cv2.waitKey(3000) # 3 saniye bekle
cv2.destroyAllWindows() # pencereleri kapa

