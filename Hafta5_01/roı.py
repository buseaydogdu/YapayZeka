# import cv2

# img1 = cv2.imread('images/squirrel11.jpg')
# # roi = img1[y1:y2, x1:x2]
# # roi = img1[200:400, 200:400] # sol göz
# roi = img1[400:520, 400:650] # burun

# cv2.imshow('parça',roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2

img1 = cv2.imread('images/squirrel11.jpg')
# roi = img1[y1:y2, x1:x2]
# parca = img1[200:400, 200:400] # sol göz
parca = img1[300:380, 400:550] # burun
img1[250:330, 400:550] = parca

cv2.imshow('parça',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()


