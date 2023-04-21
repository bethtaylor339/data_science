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
        processed_img = cv2.medianBlur(processed_img, 5)


        # Finding coordinates of circle to paint
        gray = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
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

        # Using (similar) coordinates to inpaint- not the exact same as I found it wasn't perfect
        mask = np.zeros_like(processed_img[:,:,0])
        center = (188, 212)
        radius = 22
        cv2.circle(mask, center, radius, 255, -1)

        # Extracting a similar section
        section = processed_img[120:160, 190:230]

        # Resize the mask to match the size of the image using interpolation
        mask_resized = cv2.resize(mask, processed_img.shape[:2][::-1], interpolation=cv2.INTER_LANCZOS4)

        # Inpaint the corrupted region in the color image using the extracted section
        inpainted_color = cv2.inpaint(processed_img, mask_resized, 3, cv2.INPAINT_TELEA)

        # Resize the section to match the size of the inpainted region
        section_resized = cv2.resize(section, inpainted_color.shape[:2][::-1], interpolation=cv2.INTER_LANCZOS4)
    
        # Replace the inpainted region with the extracted section
        processed_img = inpainted_color.copy()
        processed_img[mask_resized != 0] = section_resized[mask_resized != 0]

        ###DEWARPING AND REMOVING BORDERS
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
        
        ###SHARPENING
        # Apply Gaussian blur to the image
        blur = cv2.GaussianBlur(processed_img, (3,3), 20)

        # Create a sharpening filter
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

        # Apply the sharpening filter to the blurred image
        processed_img= cv2.filter2D(blur, -1, kernel)

        ###BRIGHTNESS AND CONTRAST- not sure how good this is
        target_mean = 120
        target_sd = 30

        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

        mean_pixel = np.mean(gray)
        sd_pixel = np.std(gray)
        scale_factor = target_sd / sd_pixel
        difference = abs(target_mean- mean_pixel)*scale_factor

        # Adjust the contrast and brightness
        processed_img = cv2.convertScaleAbs(processed_img, alpha=scale_factor, beta=difference)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, processed_img)

        print(f"Processed {filename}")
