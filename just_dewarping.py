import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

input_dir = "image-processing-files/test_images/"
output_dir = "image-processing-files/results/"


for filename in os.listdir(input_dir):
    # Check if the file is an image
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_dir, filename)
        img = cv2.imread(image_path)
        processed_img = img
        # Dewarping and removing borders
        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 10, 255, 0)       


        
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)
  
        # take the first contour
        cnt = contours[0]
        #https://www.geeksforgeeks.org/finding-minimum-enclosing-rectangle-in-opencv-python/
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box_img = cv2.drawContours(processed_img, [box], 0, (0, 0, 255), 2)

        extent = np.float32([[0,0], [255,0], [255,255], [0,255]])
        bounds = np.float32(box)

        M = cv2.getPerspectiveTransform(bounds, extent)

        # Apply the transformation to the image
        processed_img = cv2.warpPerspective(processed_img, M, (256, 256))
        
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, processed_img)

        print(f"Processed {filename}")
        
