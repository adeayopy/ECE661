# import matplotlib.pyplot as plt
# from skimage import io

# # Load the image using skimage
# image = io.imread('Harris_scale_2.0_hovde_3.jpg')  # Replace with your image path

# # Display the image using matplotlib
# plt.figure(figsize=(8, 8))
# plt.imshow(image)
# plt.axis('off')  # Turn off axis labels
# plt.title('Loaded Image')
# plt.show()
import cv2
from skimage import io

# Load the image using skimage
image = io.imread('output_filename.jpg')  # Replace with your image path

# Convert RGB image to BGR for OpenCV display
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Display the image using OpenCV
cv2.imshow('Image', image_bgr)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
