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

        processed_img = img
       
        ###INPAINTING
        # Using (similar) coordinates to inpaint
        mask = np.zeros_like(img[:,:,0])
        centre = (188, 212)
        radius = 22
        cv2.circle(mask, centre, radius, 255, -1)

        # Similar section
        section = img[120:160, 190:230]

        # Inpaint the corrupted region in the color image using the extracted section
        inpainted_color = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        # Resize the section to match the size of the inpainted region
        section_resized = cv2.resize(section, inpainted_color.shape[:2][::-1], interpolation=cv2.INTER_LANCZOS4)
    
        # Replace the inpainted region with the extracted section
        processed_img = inpainted_color.copy()
        processed_img[mask != 0] = section_resized[mask != 0]


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
        
        #median (salt and pepper noise removal) (better after dewarping and stuff)
        # processed_img = cv2.medianBlur(processed_img, 5)
        kernel = np.ones((3,3), np.uint8)
        processed_img= cv2.dilate(processed_img, kernel, iterations=1)
        ###SHARPENING
        
        
        # Create a sharpening filter
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        
        # Apply the sharpening filter to the blurred image
        processed_img= cv2.filter2D(processed_img, -1, kernel)
        processed_img = cv2.GaussianBlur(processed_img, (5,5), 30)

        

        #  #### REDUCING INTENSITY OF RED CHANNEL

        # b, g, r = cv2.split(processed_img)

        # # Reduce the intensity of the red channel
        # r = cv2.addWeighted(r, 0.8, 0, 0, 0)

        # # Merge the channels back together
        # processed_img = cv2.merge((b, g, r))
        

        # #Hist equalisation- using Lab (https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv)
        # lab= cv2.cvtColor(processed_img, cv2.COLOR_BGR2LAB)
        # l_channel, a, b = cv2.split(lab)

        # # Applying CLAHE to L-channel
        # # feel free to try different values for the limit and grid size:
        # clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        # cl = clahe.apply(l_channel)
     
        # # merge the CLAHE enhanced L-channel with the a and b channel
        # limg = cv2.merge((cl,a,b))

        # # Converting image from LAB Color model to BGR color spcae
        # processed_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
       



        ### BRIGHTNESS & CONTRAST

        # # Define the brightness and contrast adjustments
        # alpha = 2 # Contrast control (1.0-3.0)
        # beta = 50 # Brightness control (0-100)

        # # Apply the brightness and contrast adjustments
        # adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                


        ####Hist Equalisation
        # img_yuv = cv2.cvtColor(processed_img, cv2.COLOR_RGB2YUV)

        # # Apply histogram equalization to the Y channel
        # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        # # Convert the image back to the original color space
        # processed_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        # processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        # # Exponential transform to increase contrast
        # gamma = 2.0
        # inv_gamma = 1.0 / gamma
        # table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
        # # Apply the exponential transform to the image
        # processed_img = cv2.LUT(processed_img, table)


        # # #Brightness and Contrast- not sure how good this is
        # target_mean = 100
        # target_sd = 20
        # gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        # mean_pixel = np.mean(gray)
        # sd_pixel = np.std(gray)
        # scale_factor = mean_pixel/target_mean
        # #scale_factor = target_sd / sd_pixel
        # difference = (target_mean- mean_pixel)
        # # Adjust the contrast and brightness
        # processed_img = cv2.convertScaleAbs(processed_img, alpha=2, beta=difference/2)

        # Save the processed image to the output directory
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, processed_img)

        print(f"Processed {filename}")

