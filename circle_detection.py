import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

input_dir = "image-processing-files/test_images/"
for filename in os.listdir(input_dir):
    # Check if the file is an image
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_dir, filename)
        img = cv2.imread(image_path)
         # Finding coordinates of circle to paint
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Threshold to detect black circle
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        # Detect circles
        circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, minDist=50,
                                param1=200, param2=20, minRadius=0, maxRadius=0)

        if circles is not None:
            # Coordinates and radius of the circle
            x = int(circles[0][0][0])
            y = int(circles[0][0][1])
            r = int(circles[0][0][2])
            print('Circle center:', (x, y))
            print('Circle radius:', r)
        else:
            print('No circles found in the image.')