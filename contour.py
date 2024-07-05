import cv2
import numpy as np
import openslide
from skimage.transform import resize

wsi = openslide.OpenSlide("/voyager/projects/aditya/min_CLAM/wsis/44760.svs")
resized = [int(wsi.level_dimensions[1][1] * 1/2), int(wsi.level_dimensions[1][0] * 1/2)]
img = np.array(wsi.read_region((0,0), 1, wsi.level_dimensions[1]).convert("RGB"))

img = (resize(img, resized)*255).astype(np.uint8) 

# Convert the RGB image to grayscale
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
img_med = cv2.medianBlur(img_hsv[:,:,1], 7)  # Apply median blurring
print(img.shape)
# Thresholding
_, img_otsu = cv2.threshold(img_med, 20, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# Draw the contours on the original RGB image
cv2.drawContours(img, contours, -1, (0, 0, 0), 5)

# Save the image with contours
cv2.imwrite('contours_image.jpg', img)


print([c.shape for c in contours])
print("Image saved successfully.")
