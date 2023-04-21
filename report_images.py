
#importing libraries
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

input_dir = "image-processing-files/test_images/"
output_dir = "image-processing-files/"


# Image 37 because salt & pepper noise and lack of contrast.
im37= cv2.imread('image-processing-files/test_images/im37-RET112OD.jpg', cv2.IMREAD_COLOR)
# Image 23 because bright, lots of detail and lots of noise
im23= cv2.imread('image-processing-files/test_images/im23-RET015OD.jpg', cv2.IMREAD_COLOR)

# Pixel Values Histogram
for filename in os.listdir(input_dir):
    img = cv2.imread(os.path.join(input_dir, filename))
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    plt.xlabel('Pixel value')
    plt.ylabel('Count')
    plt.title('Histogram of pixel values for all images')
    plt.plot(hist)
plt.savefig('Pixel values histogram')

# Colour Values Histogram
colour = ('b','g','r')
for filename in os.listdir(input_dir):
    img = cv2.imread(os.path.join(input_dir, filename))
    if img is not None: 
        for i,col in enumerate(colour):
            histr = cv2.calcHist([img],[i],None,[255],[1,255])
            plt.plot(histr,color = col)
            plt.xlim([1,256])
            plt.xlabel('Pixel value')
            plt.ylabel('Count')
        
plt.savefig('BGR values histogram')

# Image 23 noise removal

kernel = np.ones((5,5), np.uint8)
dilation = cv2.dilate(im23, kernel, iterations=1)
opening = cv2.morphologyEx(im23, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(im23, cv2.MORPH_CLOSE, kernel)
median = cv2.medianBlur(im23, 5)
bilateral = cv2.bilateralFilter((im23),3,75,75)
gaussian = cv2.GaussianBlur(im23, (3,3), 30)
nmeans = cv2.fastNlMeansDenoisingColored(im23)
cv2.imwrite('im23_dilation.jpg', dilation)
cv2.imwrite('im23_opening.jpg', opening)
cv2.imwrite('im23_closing.jpg', closing)
cv2.imwrite('im23_median.jpg', median)
cv2.imwrite('im23_bilateral.jpg', bilateral)
cv2.imwrite('im23_gaussian.jpg', gaussian)
cv2.imwrite('im23_nmeans.jpg', nmeans)

# Image 37 noise removal

kernel = np.ones((5,5), np.uint8)
dilation = cv2.dilate(im37, kernel, iterations=1)
opening = cv2.morphologyEx(im37, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(im37, cv2.MORPH_CLOSE, kernel)
median = cv2.medianBlur(im37, 5)
bilateral = cv2.bilateralFilter((im37),3,75,75)
gaussian = cv2.GaussianBlur(im37, (3,3), 30)
nmeans = cv2.fastNlMeansDenoisingColored(im37)
cv2.imwrite('im37_dilation.jpg', dilation)
cv2.imwrite('im37_opening.jpg', opening)
cv2.imwrite('im37_closing.jpg', closing)
cv2.imwrite('im37_median.jpg', median)
cv2.imwrite('im37_bilateral.jpg', bilateral)
cv2.imwrite('im37_gaussian.jpg', gaussian)
cv2.imwrite('im23_nmeans.jpg', nmeans)

