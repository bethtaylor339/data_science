import cv2
im01= cv2.imread('test_images/im01-RET029OD.jpg', cv2.IMREAD_COLOR)
im02= cv2.imread('test_images/im02-RET031OS.jpg', cv2.IMREAD_COLOR)

#median (Removing salt and pepper noise) for validation 4
median1_3 = cv2.medianBlur(im01, 3)
cv2.imwrite("median1_3.jpg", median1_3)
median1_2 = cv2.medianBlur(im01, 3)
cv2.imwrite("median1_2.jpg", median1_2)

#nmeans for image 1 
nmeans_1 = cv2.fastNlMeansDenoisingColored(im01)
cv2.imwrite("nmeans_1.jpg", nmeans_1)

#Bilateral for image 1 
bilat_1 = cv2.bilateralFilter((im01),3,75,75)
cv2.imwrite("bilat_1.jpg", bilat_1)

#Gaussian on image 1 
gaussian_1 = cv2.GaussianBlur(im01, (3,3), 30)
cv2.imwrite("gaussian_1.jpg", gaussian_1)

#exp function
import math
def exponential_transform(image, c, alpha):
    for i in range(0, image.shape[1]):  # image width
        for j in range(0, image.shape[0]):  # image height

            # compute exponential transform

            image[j, i] = int(c * (math.pow(1 + alpha, image[j, i]) - 1))
    return image
scaled_1= cv2.resize(im01, ((int(im01.shape[1] * 0.6)),int(im01.shape[0] * 0.6)), interpolation=cv2.INTER_AREA)
gray_scaled_im01 = cv2.cvtColor(scaled_1, cv2.COLOR_BGR2GRAY)
EXP1 = exponential_transform(gray_scaled_im01, 130, 0.003)
colored_EXP1 = cv2.cvtColor(EXP1, cv2.COLOR_GRAY2BGR)
cv2.imwrite("EXPt1.png", colored_EXP1)

# gamma correction 
import numpy as np
gamma_1 = np.array(255*(1/255)**1.4,dtype='uint8')
cv2.imwrite("gamma_1.png", gamma_1)

#visualising histograms of pixel values
#Test 1
import matplotlib.pyplot as plt
#histt1 = cv2.calcHist([im01], [0], None, [256], [0,256])
#plt.plot(histt1)
#plt.show()

import cv2
import matplotlib.pyplot as plt
import os

# Define the directory containing the images
dir_path = "test_images"

# Loop over all the images in the directory
for filename in os.listdir(dir_path):
    # Read the image
    img = cv2.imread(os.path.join(dir_path, filename))

    # Perform the desired processing on the image
   # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate and plot the histogram
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    plt.plot(hist)

# Show the histogram for all images
plt.show()
