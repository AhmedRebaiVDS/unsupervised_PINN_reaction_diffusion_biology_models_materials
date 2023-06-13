import cv2
import numpy as np
import matplotlib.pyplot as plt

def inpaint_image(img_path, morph_size=(7,7)):
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Failed to read image from the given path")
        
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold the grayscale image to create a mask
    mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]
    
    # Perform morphological closing on the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Inpaint the image using the mask
    result = cv2.inpaint(img, mask, 21, cv2.INPAINT_TELEA) 
    
    # Convert the result to RGB
    image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return image

# Specify the path to the input image
img_path = './kvasirpytorch/kvasir-dataset/ulcerative-colitis/049c2045-5259-47d8-8f9e-bbbecd81789f.jpg'

# Inpaint the image
image = inpaint_image(img_path)

# Display the result
plt.imshow(image)
plt.axis('off')
