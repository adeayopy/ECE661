# importing the module 
import cv2 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# function to display the coordinates of 
# # of the points clicked on the image 
# def click_event(event, x, y, flags, params): 

# 	# checking for left mouse clicks 
# 	if event == cv2.EVENT_LBUTTONDOWN: 

# 		# displaying the coordinates 
# 		# on the Shell 
# 		print(x, ' ', y) 

# 		# displaying the coordinates 
# 		# on the image window 
# 		font = cv2.FONT_HERSHEY_SIMPLEX 
# 		cv2.putText(img, str(x) + ',' +
# 					str(y), (x,y), font, 
# 					1, (255, 0, 0), 2) 
# 		cv2.imshow('image', img) 

# 	# checking for right mouse clicks	 
# 	if event==cv2.EVENT_RBUTTONDOWN: 

# 		# displaying the coordinates 
# 		# on the Shell 
# 		print(x, ' ', y) 

# 		# displaying the coordinates 
# 		# on the image window 
# 		font = cv2.FONT_HERSHEY_SIMPLEX 
# 		b = img[y, x, 0] 
# 		g = img[y, x, 1] 
# 		r = img[y, x, 2] 
# 		cv2.putText(img, str(b) + ',' +
# 					str(g) + ',' + str(r), 
# 					(x,y), font, 1, 
# 					(255, 255, 0), 2) 
# 		cv2.imshow('image', img) 

# # driver function 
# if __name__=="__main__": 

# 	# reading the image 
# 	img = cv2.imread('img1.jpg', 1) 

# 	# displaying the image 
# 	cv2.imshow('image', img) 

# 	# setting mouse handler for the image 
# 	# and calling the click_event() function 
# 	cv2.setMouseCallback('image', click_event) 

# 	# wait for a key to be pressed to exit 
# 	cv2.waitKey(0) 

# 	# close the window 
# 	cv2.destroyAllWindows() 

#***************************************


# def click_event(event, x, y, flags, params): 
#     # Getting the scale factor and the original image dimensions from params
#     scale, original_width, original_height = params
    
#     # Checking for left mouse clicks 
#     if event == cv2.EVENT_LBUTTONDOWN: 
#         # Map the clicked coordinates back to the original image's coordinates
#         orig_x = int(x / scale)
#         orig_y = int(y / scale)
        
#         # Displaying the mapped coordinates on the Shell 
#         print(f'Original coordinates: {orig_x}, {orig_y}') 

#         # Displaying the coordinates on the image window 
#         font = cv2.FONT_HERSHEY_SIMPLEX 
#         cv2.putText(img, f'{orig_x},{orig_y}', (x,y), font, 
#                     1, (255, 0, 0), 2) 
#         cv2.imshow('image', img) 

#     # Checking for right mouse clicks	 
#     if event == cv2.EVENT_RBUTTONDOWN: 
#         # Map the clicked coordinates back to the original image's coordinates
#         orig_x = int(x / scale)
#         orig_y = int(y / scale)
        
#         # Displaying the mapped coordinates on the Shell 
#         print(f'Original coordinates: {orig_x}, {orig_y}') 

#         # Getting the color values from the original image
#         b = original_img[orig_y, orig_x, 0] 
#         g = original_img[orig_y, orig_x, 1] 
#         r = original_img[orig_y, orig_x, 2] 
        
#         # Displaying the color values on the image window 
#         font = cv2.FONT_HERSHEY_SIMPLEX 
#         cv2.putText(img, f'{b},{g},{r}', (x,y), font, 1, (255, 255, 0), 2) 
#         cv2.imshow('image', img) 

# # Driver function 
# if __name__ == "__main__": 

#     # Reading the original image 
#     original_img = cv2.imread('alex_honnold.jpg', 1)
    
#     # Get dimensions of the original image
#     original_height, original_width = original_img.shape[:2]

#     # Set a desired width and height for the image
#     max_width = 800  # Set this to your screen width or desired width
#     max_height = 600 # Set this to your screen height or desired height

#     # Calculate the scaling factor
#     scale_width = max_width / original_width
#     scale_height = max_height / original_height
#     scale = min(scale_width, scale_height)

#     # Calculate the new dimensions for display
#     new_width = int(original_width * scale)
#     new_height = int(original_height * scale)

#     # Resize the image for display
#     img = cv2.resize(original_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

#     # Displaying the resized image 
#     cv2.imshow('alex_honnold.jpg', img)

#     # Setting mouse handler for the image and passing the scaling factor and original dimensions
#     cv2.setMouseCallback('alex_honnold.jpg', click_event, param=(scale, original_width, original_height))

#     # Wait for a key to be pressed to exit 
#     cv2.waitKey(0) 

#     # Close the window 
#     cv2.destroyAllWindows()


#***************************************

# # Load an image from a file (replace with your image path)
# img = mpimg.imread('img1.jpg')

# # Display the image
# plt.imshow(img, origin='upper')

# # Show the plot with the ability to hover and see coordinates
# plt.show()
##***************************************

# Points in the source image (from where the ROI is taken)
pts_src = np.array([[97, 638], [135, 74], [741, 100], [720, 630]])

# Corresponding points in the destination image
# img1
pts_dst = np.array([[678, 3198], [416, 833], [2567, 887], [2425, 2298]])
# img2
# pts_dst = np.array([[1402, 2496], [813, 1132], [2826, 1040], [2670, 2539]])
# img3
# pts_dst = np.array([[1394, 1934], [1467, 1484], [2106, 1582], [2130, 1966]])

# Set up matrix A as shown above
A = []
for i in range(4):
    x, y = pts_src[i][0], pts_src[i][1]
    x_prime, y_prime = pts_dst[i][0], pts_dst[i][1]
    A.append([x, y, 1, 0, 0, 0, -x_prime*x, -x_prime*y, -x_prime])
    A.append([0, 0, 0, x, y, 1, -y_prime*x, -y_prime*y, -y_prime])

A = np.array(A)

# Use SVD to solve the system of equations
U, S, V = np.linalg.svd(A)
H = V[-1, :].reshape(3, 3)

# Apply the homography manually
def apply_homography(H, src_image, dst_image):
    height, width = dst_image.shape[:2]
    for i in range(src_image.shape[0]):
        for j in range(src_image.shape[1]):
            point = np.array([i, j, 1])
            mapped_point = np.dot(H, point)
            mapped_point /= mapped_point[2]
            x, y = int(mapped_point[0]), int(mapped_point[1])
            if 0 <= x < height and 0 <= y < width:
                dst_image[x, y] = src_image[i, j]
    return dst_image

# Load images
src_image = cv2.imread('alex_honnold.jpg')
dst_image = cv2.imread('img1.jpg')

# Apply the homography to project the ROI
warped_image = apply_homography(H, src_image, dst_image)

# Show the resulting image
plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
plt.show()

