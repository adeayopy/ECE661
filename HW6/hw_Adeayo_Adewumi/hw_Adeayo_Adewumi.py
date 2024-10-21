# %%
# Help was used from ChatGPT AND previous assignments (Best solution 2 - Wei Xu)
import numpy as np
import cv2 

# %%

# Function to compute histogram of an image
def compute_histogram(image):
    hist = np.zeros(256, dtype=int)
    for pixel in image.flatten():
        hist[pixel] += 1
    return hist

# Function to normalize the histogram
def normalize_histogram(hist, total_pixels):
    return hist / total_pixels

# Function to compute cumulative sums and means
def compute_cumulative_sums_and_means(hist):
    cum_sum = np.cumsum(hist)
    cum_mean = np.cumsum(np.arange(256) * hist)
    return cum_sum, cum_mean 

def refine_foreground_segmentation(image, thresh, optimal_threshold):

    # Extract the preliminary foreground using the initial threshold
    foreground_pixels = image[thresh == 255]
    
    # Apply Otsu's threshold to the foreground pixels to refine the thresholding
    if len(foreground_pixels) > 0:
        refined_threshold = otsu_threshold(foreground_pixels)[0]
    else:
        refined_threshold = optimal_threshold  # Use the original threshold if no foreground is detected
    
    # Create the final refined mask
    # Use the refined threshold on the original image, but only where the initial foreground was detected
    refined_mask = np.zeros_like(image, dtype=np.uint8)
    refined_mask[(image > refined_threshold) & (thresh == 255)] = 255
    
    return refined_mask



def combine_masks_with_bitwise_and(masks):
    # Start with the first mask
    combined_mask = masks[0]
    # Apply bitwise AND with subsequent masks
    for mask in masks[1:]:
        combined_mask = np.bitwise_and(combined_mask, mask)
    
    return combined_mask

# Function to find optimal threshold using Otsu's method
def otsu_threshold(image):
    # Compute histogram
    hist = compute_histogram(image)
    # Normalize the histogram
    total_pixels = image.size
    prob_hist = normalize_histogram(hist, total_pixels)
    # Compute cumulative sums and means
    cum_sum, cum_mean = compute_cumulative_sums_and_means(prob_hist)
    # Compute global mean
    global_mean = cum_mean[-1]
    # Initialize variables to find optimal threshold
    max_variance = -1
    optimal_threshold = 0
    for t in range(256):
        # Probabilities of two classes (background and foreground)
        w0 = cum_sum[t]  # Background class
        w1 = 1 - w0      # Foreground class
        if w0 == 0 or w1 == 0:
            continue  # Avoid division by zero
        # Means of the two classes
        mean0 = cum_mean[t] / w0
        mean1 = (cum_mean[-1] - cum_mean[t]) / w1
        # Compute between-class variance
        between_class_variance = w0 * w1 * (mean0 - mean1) ** 2
        # Update the optimal threshold if the variance is maximal
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            optimal_threshold = t
    # Threshold the image using the optimal threshold
    thresh = np.where(image > optimal_threshold, 255, 0).astype(np.uint8)
    
    return optimal_threshold, thresh





# %%
def compute_texture_features(image, N):

    pad = N // 2  # Padding size for border handling
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)
    texture_map = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + N, j:j + N]
            mean_intensity = np.mean(window)
            variance = np.mean((window - mean_intensity) ** 2)
            texture_map[i, j] = variance

    texture_map = cv2.normalize(texture_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return texture_map.astype(np.uint8)

def iterative_otsu(feature_maps, iterations_list):

    refined_masks = []
    
    for feature_map, iterations in zip(feature_maps, iterations_list):
        initial_threshold, initial_mask = otsu_threshold(feature_map)
        refined_mask = initial_mask
        
        # Perform the iterative refinement
        for _ in range(iterations):
            refined_mask = refine_foreground_segmentation(feature_map, refined_mask, initial_threshold)
        
        refined_masks.append(refined_mask)
    
    return refined_masks

def extract_texture_features(image, window_sizes):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature_maps = [compute_texture_features(grayscale_image, N) for N in window_sizes]
    return feature_maps

def save_masks(refined_masks, window_sizes, image_name):
    for idx, mask in enumerate(refined_masks):
        filename = f'{image_name}_refined_mask_window_{window_sizes[idx]}.png'
        cv2.imwrite(filename, mask)
        print(f"Saved {filename}")




# %%


def extract_contours(binary_mask):
    contours = []
    visited = set()  # To keep track of visited points in the contour
    # Define the 8-connected neighbors relative positions (dy, dx)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals
    # Loop through each pixel in the binary mask
    for y in range(1, binary_mask.shape[0] - 1):
        for x in range(1, binary_mask.shape[1] - 1):
            # Check if the current pixel is part of the foreground
            if binary_mask[y, x] == 1:
                # Check if the current pixel has at least one neighboring background pixel
                is_boundary = any(
                    binary_mask[y + dy, x + dx] == 0
                    for dy, dx in neighbors
                    if 0 <= y + dy < binary_mask.shape[0] and 0 <= x + dx < binary_mask.shape[1]
                )
                # If it is a boundary pixel and not visited, add it to the contour
                if is_boundary and (y, x) not in visited:
                    contour = []  # Store the current contour points
                    stack = [(y, x)]  # Use a stack for DFS-based contour tracing
                    while stack:
                        current_y, current_x = stack.pop()
                        # Mark the current point as visited
                        visited.add((current_y, current_x))
                        contour.append((current_x, current_y))  # Store the contour point (x, y)
                        # Explore neighbors to continue tracing the boundary
                        for dy, dx in neighbors:
                            ny, nx = current_y + dy, current_x + dx
                            # Ensure the neighbor is within the bounds of the image
                            if 0 <= ny < binary_mask.shape[0] and 0 <= nx < binary_mask.shape[1]:
                                # Check if the neighbor is a boundary point
                                if binary_mask[ny, nx] == 1 and (ny, nx) not in visited:
                                    # Verify that this point is on the boundary
                                    if any(
                                        binary_mask[ny + ddy, nx + ddx] == 0
                                        for ddy, ddx in neighbors
                                        if 0 <= ny + ddy < binary_mask.shape[0] and 0 <= nx + ddx < binary_mask.shape[1]
                                    ):
                                        stack.append((ny, nx))
                    if contour:
                        contours.append(contour)
    return contours

def draw_and_save_contours(contours, image_shape, output_filename):
    # Create a blank image to draw contours (black background)
    contour_image = np.zeros(image_shape, dtype=np.uint8)

    # Draw each contour on the image with white color (255)
    for contour in contours:
        for (x, y) in contour:
            contour_image[y, x] = 255  # Set pixel to white for the contour

    # Save the contour image
    cv2.imwrite(output_filename, contour_image)
    print(f"Contour image saved as {output_filename}")




# %%

# # # Load an RGB image and split into channels
# # dog_image = cv2.imread('pics/dog_small.jpg')
# # R, G, B = cv2.split(dog_image)
# # # Perform iterative otsu
# # dog_refined_rgb_masks = iterative_otsu([R,G,B], [3, 5, 5])
# # # Combine the masks using bitwise AND
# # dog_final_rgb_mask = combine_masks_with_bitwise_and(dog_refined_rgb_masks)
# # # normlized_rgb_mask=normalize_binary_mask(refined_rgb_masks)
# # # Save each RGB mask
# # save_masks(dog_refined_rgb_masks, ['R','G','B'], 'dog')
# # cv2.imwrite('dog_refined_mask.png', dog_final_rgb_mask)

# # dog_final_rgb_mask = dog_final_rgb_mask // 255
# # # Extract contours
# # contours = extract_contours(dog_final_rgb_mask)
# # # Save contours as an image
# # draw_and_save_contours(contours, dog_final_rgb_mask.shape, 'dog_contour_RGB.png')



# # Extract texture-based features
# # dog_feature_maps = extract_texture_features(dog_image, [11, 13, 17])
# # # Perform iterative otsu on texture maps and normalize
# # dog_refined_text_masks = iterative_otsu(dog_feature_maps, [1, 1, 1])
# # # refined_masks=normalize_binary_mask(refined_masks)
# # # Save each refined mask

# # dog_final_text_mask = combine_masks_with_bitwise_and(dog_refined_text_masks)
# # save_masks(dog_refined_text_masks, [11, 13, 17], 'dog')
# # cv2.imwrite('dog_final_texture_based_mask.png', dog_final_text_mask)


# # dog_final_text_mask = dog_final_text_mask // 255
# # # Extract contours
# # contours = extract_contours(dog_final_text_mask)
# # # Save contours as an image
# # draw_and_save_contours(contours, dog_final_text_mask.shape, 'dog_contour_txt.png')


# #*******************************************************************************************************

# # Load an RGB image and split into channels
# # flower_image = cv2.imread('pics/flower_small.jpg')
# # R, G, B = cv2.split(flower_image)
# # # Perform iterative otsu
# # flower_refined_rgb_masks = iterative_otsu([R,G,B], [1, 1, 1])
# # # Combine the masks using bitwise AND
# # flower_final_rgb_mask = combine_masks_with_bitwise_and(flower_refined_rgb_masks)
# # # normlized_rgb_mask=normalize_binary_mask(refined_rgb_masks)
# # # Save each RGB mask
# # save_masks(flower_refined_rgb_masks, ['R','G','B'], 'flower')
# # cv2.imwrite('flower_refined_mask_RGB.png', flower_final_rgb_mask)

# # flower_final_rgb_mask = flower_final_rgb_mask // 255
# # # Extract contours
# # contours = extract_contours(flower_final_rgb_mask)
# # # Save contours as an image
# # draw_and_save_contours(contours, flower_final_rgb_mask.shape, 'flower_contour_RGB.png')

# # Extract texture-based features
# # flower_feature_maps = extract_texture_features(flower_image, [15, 17, 21])
# # # Perform iterative otsu on texture maps and normalize
# # flower_refined_text_masks = iterative_otsu(flower_feature_maps, [1, 1, 1])

# # flower_final_text_mask = combine_masks_with_bitwise_and(flower_refined_text_masks)
# # save_masks(flower_refined_text_masks, [15, 17, 21], 'flower')
# # cv2.imwrite('flower_final_text_based_mask.png', flower_final_text_mask)

# # flower_final_text_mask = flower_final_text_mask // 255
# # # Extract contours
# # contours = extract_contours(flower_final_text_mask)
# # # Save contours as an image
# # draw_and_save_contours(contours, flower_final_text_mask.shape, 'flower_contour_txt.png')


# #*******************************************************************************************************

# # Load an RGB image and split into channels
# car_image = cv2.imread('pics/car.jpg')
# # R, G, B = cv2.split(car_image)
# # # Perform iterative otsu
# # car_refined_rgb_masks = iterative_otsu([R,G,B], [1, 1, 1])
# # # Combine the masks using bitwise AND
# # car_final_rgb_mask = combine_masks_with_bitwise_and(car_refined_rgb_masks)
# # # normlized_rgb_mask=normalize_binary_mask(refined_rgb_masks)
# # # Save each RGB mask
# # save_masks(car_refined_rgb_masks, ['R','G','B'], 'car')
# # cv2.imwrite('car_refined_mask_RGB.png', car_final_rgb_mask)

# # car_final_rgb_mask = car_final_rgb_mask // 255
# # # Extract contours
# # contours = extract_contours(car_final_rgb_mask)
# # # Save contours as an image
# # draw_and_save_contours(contours, car_final_rgb_mask.shape, 'car_contour_RGB.png')

# # Extract texture-based features
# # car_feature_maps = extract_texture_features(car_image, [7, 9, 13])
# # # Perform iterative otsu on texture maps and normalize
# # car_refined_text_masks = iterative_otsu(car_feature_maps, [1, 2, 1])

# # car_final_text_mask = combine_masks_with_bitwise_and(car_refined_text_masks)
# # save_masks(car_refined_text_masks, [7, 9, 13], 'car')
# # cv2.imwrite('car_final_text_based_mask.png', car_final_text_mask)

# # car_final_text_mask = car_final_text_mask // 255
# # # Extract contours
# # contours = extract_contours(car_final_text_mask)
# # # Save contours as an image
# # draw_and_save_contours(contours, car_final_text_mask.shape, 'car_contour_txt.png')


# #*******************************************************************************************************

# # Load an RGB image and split into channels
bottle_image = cv2.imread('pics/bottle.jpg')
# # R, G, B = cv2.split(bottle_image)
# # # Perform iterative otsu
# # bottle_refined_rgb_masks = iterative_otsu([R,G,B], [1, 1, 1])
# # # Combine the masks using bitwise AND
# # bottle_final_rgb_mask = combine_masks_with_bitwise_and(bottle_refined_rgb_masks)
# # # normlized_rgb_mask=normalize_binary_mask(refined_rgb_masks)
# # # Save each RGB mask
# # save_masks(bottle_refined_rgb_masks, ['R','G','B'], 'bottle')
# # cv2.imwrite('bottle_refined_mask_RGB.png', bottle_final_rgb_mask)

# # bottle_final_rgb_mask = bottle_final_rgb_mask // 255
# # # Extract contours
# # contours = extract_contours(bottle_final_rgb_mask)
# # # Save contours as an image
# # draw_and_save_contours(contours, bottle_final_rgb_mask.shape, 'bottle_contour_RGB.png')

# Extract texture-based features
bottle_feature_maps = extract_texture_features(bottle_image, [21, 25, 29])
# Perform iterative otsu on texture maps and normalize
bottle_refined_text_masks = iterative_otsu(bottle_feature_maps, [1, 1, 1])

bottle_final_text_mask = combine_masks_with_bitwise_and(bottle_refined_text_masks)
save_masks(bottle_refined_text_masks, [21, 25, 29], 'bottle')
cv2.imwrite('bottle_final_text_based_mask.png', bottle_final_text_mask)

bottle_final_text_mask = bottle_final_text_mask // 255
# Extract contours
contours = extract_contours(bottle_final_text_mask)
# Save contours as an image
draw_and_save_contours(contours, bottle_final_text_mask.shape, 'bottle_contour_txt.png')




