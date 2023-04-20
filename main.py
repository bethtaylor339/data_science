import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Set the input and output directories
input_dir = "image-processing-files/test_images/"
output_dir = "image-processing-files/results/"


# Plotting pixel values of all images
for filename in os.listdir(input_dir):
    img = cv2.imread(os.path.join(input_dir, filename))
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    plt.xlabel('Pixel value')
    plt.ylabel('Count')
    plt.title('Histogram of pixel values for all images')
    plt.plot(hist)
plt.savefig('Pixel values histogram')

#Plotting bgr values of all images
color = ('b','g','r')
for filename in os.listdir(input_dir):
    img = cv2.imread(os.path.join(input_dir, filename))
    if img is not None: 
        for i,col in enumerate(color):
            histr = cv2.calcHist([img],[i],None,[255],[1,255])
            plt.plot(histr,color = col)
            plt.xlim([1,256])
        
plt.savefig('BGR values histogram')


for filename in os.listdir(input_dir):
    # Check if the file is an image
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_dir, filename)
        img = cv2.imread(image_path)

        # Image processing

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

        # Using (similar) coordinates to inpaint- not the exact same as I found it wasn't perfect
        mask = np.zeros_like(img[:,:,0])
        center = (188, 212)
        radius = 22
        cv2.circle(mask, center, radius, 255, -1)

        # Extracting a similar section
        section = img[128:162, 188:232]

        # Resize the mask to match the size of the image using interpolation
        mask_resized = cv2.resize(mask, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

        # Inpaint the corrupted region in the color image using the extracted section
        inpainted_color = cv2.inpaint(img, mask_resized, 3, cv2.INPAINT_NS)

        # Resize the section to match the size of the inpainted region
        section_resized = cv2.resize(section, inpainted_color.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        cv2.imshow('image', section_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Replace the inpainted region with the extracted section
        output = inpainted_color.copy()
        output[mask_resized != 0] = section_resized[mask_resized != 0]

        # Display the output image
        cv2.imshow('image', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



        # Save the processed image to the output directory
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, processed_img)

        print(f"Processed {filename}")
