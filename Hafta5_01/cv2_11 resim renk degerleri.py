import cv2
resim=cv2.imread("images/renklitop.jpg")

x,y,r = resim.shape
print( resim.shape)
for a in range (x):
    for b in range(y):
        print(f"x,y değeri {resim[a],[b]}")
        if resim[a][b][2] > 200 : resim[a][b][2] - [0]

cv2.imshow("Resim",resim)       
cv2.waitKey(0)