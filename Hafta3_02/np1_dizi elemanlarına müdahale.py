# boş resim oluşturma
import cv2, random
import numpy as np
max = 200
m = random.randint(0,255);
y = random.randint(0,255); k = random.randint(0,255);
r1= np.full((max, max, 3), [m, y, k], dtype=np.uint8)

# r1[max//2,max//2] = [0,0,0]
for a in range(max): r1[max//2,a] = [0,0,0]
for a in range(max): r1[a,max//2] = [255,255,255]

# for a in range (max):
#     for b in range(a, max):
#       if a == b : r1[a,b] = [0, 0, 0]
#       if a == b//2 : r1[a,b] = [0, 0, 255]

# for c in range(100):
#    m1=random.randint(0,255); y1=random.randint(0,255); k1=random.randint(0,255);
#    x = random.randint(3,max-3); y = random.randint(3,max-3)
#    r1[x-1, y-1] = [m1, y1, k1]
#    r1[x-1, y] = [m1, y1, k1]
#    r1[x-1, y+1] = [m1, y1, k1]
#    r1[x, y-1] = [m1, y1, k1]
#    r1[x, y] = [m1, y1, k1]
#    r1[x, y+1] = [m1, y1, k1]
#    r1[x+1, y-1] = [m1, y1, k1]
#    r1[x+1, y] = [m1, y1, k1]
#    r1[x+1, y+1] = [m1, y1, k1]

cv2.imshow("Resim", r1)
cv2.waitKey(0) # Herhangi bir tuşa basınca veya 5 saniye sonra kapat.
cv2.destroyAllWindows()