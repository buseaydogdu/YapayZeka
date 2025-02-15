# boş resim oluşturma
# pip install opencv-python

import cv2
import numpy as np

# aşağıdaki [255, 255, 255] rakamlarını oynayarak ne olduğuna bakın
r1= np.full((250, 250, 3), [210, 710, 210], dtype=np.uint8)
print(r1)

cv2.imshow("Olusan resim", r1)

cv2.waitKey(0)
 
# cv2.destroyAllWindows()
