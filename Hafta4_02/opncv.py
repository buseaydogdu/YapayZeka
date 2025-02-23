import cv2, numpy as np
# img = cv2.imread('resimler/squirrel1.jpg')
img = cv2.imread('images/squirrel11.jpg')
print(img)
cv2.imshow('deneme',img)
newarr = np.array_split(img, 3)
print("\nBölünmedeki parçaları alma")
cv2.imshow('1.parça',newarr[0])
cv2.imshow('2.parça',newarr[1])
cv2.imshow('3.parça',newarr[2])

cv2.waitKey(0) # tuşa basılana kadar bekle
cv2.destroyAllWindows() # pencereleri kapa
