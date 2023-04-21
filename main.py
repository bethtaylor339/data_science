import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Set the input and output directories
input_dir = "image-processing-files/test_images/"
output_dir = "image-processing-files/results/"


for filename in os.listdir(input_dir):
    # Check if the file is an image
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_dir, filename)
        img = cv2.imread(image_path)

        # Image processing

        # Using (similar) coordinates to inpaint- not the exact same as I found it wasn't perfect
        mask = np.zeros_like(img[:,:,0])
        centre = (188, 212)
        radius = 22
        cv2.circle(mask, centre, radius, 255, -1)

        # Extracting a similar section
        section = img[120:160, 190:230]

        # Resize the mask to match the size of the image using interpolation
        mask_resized = cv2.resize(mask, img.shape[:2][::-1], interpolation=cv2.INTER_LANCZOS4)

        # Inpaint the corrupted region in the color image using the extracted section
        inpainted_color = cv2.inpaint(img, mask_resized, 3, cv2.INPAINT_TELEA)

        # Resize the section to match the size of the inpainted region
        section_resized = cv2.resize(section, inpainted_color.shape[:2][::-1], interpolation=cv2.INTER_LANCZOS4)
    
        # Replace the inpainted region with the extracted section
        processed_img = inpainted_color.copy()
        processed_img[mask_resized != 0] = section_resized[mask_resized != 0]


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
        
        #median (salt and pepper noise removal)
        processed_img = cv2.medianBlur(processed_img, 5)
        processed_img = cv2.GaussianBlur(processed_img, (5,5), 10)

        #Hist Equalisation
        img_yuv = cv2.cvtColor(processed_img, cv2.COLOR_RGB2YUV)

        # Apply histogram equalization to the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        # Convert the image back to the original color space
        processed_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        # # Exponential transform to increase contrast
        # gamma = 3.0
        # inv_gamma = 1.0 / gamma
        # table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
        # # Apply the exponential transform to the image
        # processed_img = cv2.LUT(processed_img, table)


        # #Brightness and Contrast- not sure how good this is
        # target_mean = 120

        # gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        # mean_pixel = np.mean(gray)
        # scale_factor = target_mean / mean_pixel
        # difference = abs(target_mean- mean_pixel)
        # # Adjust the contrast and brightness
        # processed_img = cv2.convertScaleAbs(processed_img, alpha=1, beta=-10)

        # Save the processed image to the output directory
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, processed_img)

        print(f"Processed {filename}")

