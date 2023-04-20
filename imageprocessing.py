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
       
        processed_img = np.array(255*(1/255)**1.4,dtype='uint8')
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, processed_img)

        print(f"Processed {filename}")
