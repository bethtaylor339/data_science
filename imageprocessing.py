import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

input_dir = "image-processing-files/test_images/"
output_dir = "image-processing-files/results/"
for filename in os.listdir(input_dir):
    # Check if the file is an image
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_dir, filename)
        img = cv2.imread(image_path)

        #median (salt and pepper noise removal)
        processed_img = img
        kernel = np.ones((3,3), np.uint8)
        # processed_img= cv2.dilate(processed_img, kernel, iterations=1)
        processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel)
        # closing = cv2.morphologyEx(im23, cv2.MORPH_CLOSE, kernel)
        # # processed_img = cv2.medianBlur(processed_img, 5)

        # mask = np.zeros_like(processed_img[:,:,0])
        # center = (190, 225)
        # radius = 22
        # mask = cv2.circle(mask, center, radius, 255, -1)

        # processed_img = cv2.inpaint(processed_img, mask, radius, cv2.INPAINT_TELEA)
        # ###DEWARPING AND REMOVING BORDERS
        # # Dewarping and removing borders
        # gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(gray, 10, 255, 0)       


        
        # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
        #                        cv2.CHAIN_APPROX_SIMPLE)
  
        # # take the first contour
        # cnt = contours[0]
        # #https://www.geeksforgeeks.org/finding-minimum-enclosing-rectangle-in-opencv-python/
        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # box_img = cv2.drawContours(processed_img, [box], 0, (0, 0, 255), 2)

        # extent = np.float32([[0,0], [255,0], [255,255], [0,255]])
        # bounds = np.float32(box)

        # M = cv2.getPerspectiveTransform(bounds, extent)

        # # Apply the transformation to the image
        # processed_img = cv2.warpPerspective(processed_img, M, (256, 256))

    
            
        # ###SHARPENING
        # # Apply Gaussian blur to the image
        # blur = cv2.GaussianBlur(processed_img, (3,3), 20)

        # # Create a sharpening filter
        # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

        # # Apply the sharpening filter to the blurred image
        # processed_img= cv2.filter2D(blur, -1, kernel)

        ###BRIGHTNESS AND CONTRAST- not sure how good this is
        # target_mean = 120
        # target_sd = 30

        # gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

        # mean_pixel = np.mean(gray)
        # sd_pixel = np.std(gray)
        # scale_factor = target_sd / sd_pixel
        # difference = abs(target_mean- mean_pixel)*scale_factor

        # # Adjust the contrast and brightness
        # processed_img = cv2.convertScaleAbs(processed_img, alpha=scale_factor, beta=difference)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, processed_img)

        print(f"Processed {filename}")
