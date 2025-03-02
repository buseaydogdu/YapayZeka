import cv2, random, numpy 

video = cv2.VideoCapture(0)
while video.isOpened():

    ret, img = video.read()
# Setting All parameters
    t_lower = 100 # Lower Threshold
    t_upper = 200 # Upper threshold
    aperture_size = 3 # Aperture size
    img=cv2.resize(img, (750, 500))

# Applying the Canny Edge filter
# with Custom Aperture Size
edge = cv2.Canny(img, t_lower, t_upper,
                apertureSize=aperture_size)
cv2.imshow('original', cv2.pyrDown(img))
cv2.imshow('edge', cv2.pyrDown(img))
cv2.waitKey(0)
cv2.destroyAllWindows()

