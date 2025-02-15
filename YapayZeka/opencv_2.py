# boş resim oluşturma
import cv2
import numpy as np

for a in range(255):
    r1= np.full((200, 300, 3), [10, a, a], dtype=np.uint8) 
    cv2.imshow("Ak", r1)
    cv2.waitKey(10)
 
cv2.waitKey(0)
cv2.destroyAllWindows()

