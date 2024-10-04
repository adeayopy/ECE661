# %%
# Some help was used from ChatGPT for code optimization and commenting
import cv2
import numpy as np
import random
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# %%
def compute_homography(p1, p2):
    # Calculate homography matrix using the Direct Linear Transform (DLT)
    A = []
    for i in range(len(p1)):
        x1, y1 = p1[i][0], p1[i][1]
        x2, y2 = p2[i][0], p2[i][1]
        A.append([-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])
    A = np.array(A)
    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    return H / H[2, 2]  # Normalize the homography


# %%

def apply_homography(H, pt):
    # Apply homography to a point
    pt = np.array([pt[0], pt[1], 1.0])
    transformed_pt = np.dot(H, pt)
    transformed_pt /= transformed_pt[2]  # Normalize
    return transformed_pt[0], transformed_pt[1]



# %%
def ransac_outlier_rejection(pts1, pts2, threshold=5.0, max_iterations=1000):
    best_H = None
    max_inliers = 0
    best_inliers = []
    best_outliers = []

    for _ in range(max_iterations):
        # Step 1: Randomly select 4 points
        indices = random.sample(range(len(pts1)), 4)
        p1_sample = [pts1[i] for i in indices]
        p2_sample = [pts2[i] for i in indices]

        # Step 2: Compute homography based on the sample
        H = compute_homography(p1_sample, p2_sample)

        inliers = []
        outliers = []

        # Step 3: Compute reprojection error for all points
        for i in range(len(pts1)):
            projected_pt = apply_homography(H, pts1[i])
            error = np.linalg.norm(np.array(projected_pt) - np.array(pts2[i]))

            if error < threshold:
                inliers.append(i)
            else:
                outliers.append(i)

        # Step 4: Keep the model with the most inliers
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_H = H
            best_inliers = inliers
            best_outliers = outliers

    return best_H, best_inliers, best_outliers



# %%
def plot_inliers_outliers(img1, img2, keypoints1, keypoints2, inliers, outliers, output_name):
    # Create an image combining the two images side by side
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = img1
    combined_img[:h2, w1:] = img2

    # Plot inliers in blue and outliers in red
    for i in inliers:
        pt1 = np.int32(keypoints1[i])
        pt2 = np.int32(keypoints2[i]) + np.array([w1, 0])  # Shift pt2 horizontally
        cv2.line(combined_img, tuple(pt1), tuple(pt2), (0, 0, 255), 1)
        cv2.circle(combined_img, tuple(pt1), 5, (255, 0, 0), -1)
        cv2.circle(combined_img, tuple(pt2), 5, (255, 0, 0), -1)

    for i in outliers:
        pt1 = np.int32(keypoints1[i])
        pt2 = np.int32(keypoints2[i]) + np.array([w1, 0])  # Shift pt2 horizontally
        cv2.line(combined_img, tuple(pt1), tuple(pt2), (0, 0, 255), 1)
        cv2.circle(combined_img, tuple(pt1), 5, (0, 0, 255), -1)
        cv2.circle(combined_img, tuple(pt2), 5, (0, 0, 255), -1)

    cv2.imwrite(output_name, combined_img)




# %%
def compute_homography_from_inliers(pts1, pts2, inliers):
    # Extract the inlier points from the original point sets
    inlier_pts1 = [pts1[i] for i in inliers]
    inlier_pts2 = [pts2[i] for i in inliers]

    # Compute homography using the inlier points
    H_inliers = compute_homography(inlier_pts1, inlier_pts2)

    return H_inliers, inlier_pts1, inlier_pts2


# def reprojection_error(h, pts1, pts2):
#     """Computes the reprojection error between projected points and actual points."""
#     H = h.reshape((3, 3))  # Reshape h into the 3x3 homography matrix
#     total_error = []
    
#     for i in range(len(pts1)):
#         pt1 = pts1[i]
#         pt2 = pts2[i]
#         pt1_proj = apply_homography(H, pt1)
#         error = np.linalg.norm(pt2 - pt1_proj)  # Euclidean distance error
#         total_error.append(error)
    
#     return np.array(total_error)


def refine_homography(H_init, pts1, pts2):
    """Refine homography using nonlinear least squares optimization."""
    h_init = H_init.flatten()  # Flatten the 3x3 homography matrix to a 9-element vector

    # Use Levenberg-Marquardt optimization to minimize reprojection error
    result = least_squares(reprojection_error, h_init, args=(pts1, pts2))

    # Reshape the optimized homography back into a 3x3 matrix
    H_refined = result.x.reshape((3, 3))

    # Normalize the homography matrix
    H_refined /= H_refined[2, 2]

    return H_refined

# # Extracredit
def reprojection_error(h, pts1, pts2):
    """Calculate the reprojection error between pts1 and pts2 given homography parameters."""
    # Reshape h back into a 3x3 matrix
    H = h.reshape(3, 3)

    # Apply homography to pts1
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    projected_pts = (H @ pts1_homo.T).T

    # Normalize to convert from homogeneous to Cartesian coordinates
    projected_pts /= projected_pts[:, 2][:, np.newaxis]

    # Reprojection error (difference between projected points and actual points)
    error = projected_pts[:, :2] - pts2

    return error.flatten()

# Extracredit
def jacobian(h, pts1):
    """Calculate the Jacobian matrix of the reprojection error with respect to h."""
    H = h.reshape(3, 3)
    num_points = pts1.shape[0]
    J = np.zeros((2 * num_points, 9))

    for i in range(num_points):
        x, y = pts1[i, 0], pts1[i, 1]

        # Project the point using the homography
        denom = (H[2, 0] * x + H[2, 1] * y + H[2, 2])
        x_prime = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / denom
        y_prime = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / denom

        # Derivatives of the projected point with respect to h
        J[2 * i, 0] = x / denom
        J[2 * i, 1] = y / denom
        J[2 * i, 2] = 1 / denom
        J[2 * i, 6] = -x_prime * x / denom
        J[2 * i, 7] = -x_prime * y / denom
        J[2 * i, 8] = -x_prime / denom

        J[2 * i + 1, 3] = x / denom
        J[2 * i + 1, 4] = y / denom
        J[2 * i + 1, 5] = 1 / denom
        J[2 * i + 1, 6] = -y_prime * x / denom
        J[2 * i + 1, 7] = -y_prime * y / denom
        J[2 * i + 1, 8] = -y_prime / denom

    return J

# Extracredit
def refine_homography_lm(H_init, pts1, pts2, max_iters=100, lambda_init=1e-3, tol=1e-6):
    """Refine the homography using the Levenberg-Marquardt algorithm."""
    # Convert pts1 and pts2 to NumPy arrays if they are not already
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    h = H_init.flatten()  # Flatten the initial homography to a vector
    lambda_param = lambda_init
    prev_error = np.inf

    for iteration in range(max_iters):
        # Compute reprojection error
        error = reprojection_error(h, pts1, pts2)
        error_norm = np.linalg.norm(error)

        if error_norm < tol:
            print(f"Converged after {iteration} iterations")
            break

        # Compute the Jacobian
        J = jacobian(h, pts1)

        # Compute the update: (J^T J + lambda I) delta_h = J^T error
        JTJ = J.T @ J
        JTe = J.T @ error
        delta_h = np.linalg.inv(JTJ + lambda_param * np.eye(9)) @ JTe

        # Update h
        h_new = h - delta_h

        # Compute new reprojection error
        new_error = reprojection_error(h_new, pts1, pts2)
        new_error_norm = np.linalg.norm(new_error)

        # Check if the new error is smaller
        if new_error_norm < error_norm:
            # Accept the new solution
            h = h_new
            prev_error = new_error_norm
            lambda_param /= 10  # Reduce lambda (more Gauss-Newton)
        else:
            # Reject the new solution and increase lambda (more gradient descent)
            lambda_param *= 10

        # Check for convergence based on error difference
        if abs(prev_error - new_error_norm) < tol:
            print(f"Converged after {iteration} iterations with error {new_error_norm}")
            break

    # Reshape the refined homography back into 3x3 matrix
    H_refined = h.reshape(3, 3)
    H_refined /= H_refined[2, 2]  # Normalize the homography

    return H_refined



def compute_average_error(H, pts1, pts2):
    """Computes the average reprojection error using the given homography."""
    projected_pts = []
    for pt in pts1:
        projected_pt = apply_homography(H, pt)
        projected_pts.append(projected_pt)

    projected_pts = np.array(projected_pts)
    error = np.linalg.norm(pts2 - projected_pts, axis=1)  # Euclidean distance for each point
    avg_error = np.mean(error)  # Average reprojection error
    
    return avg_error


# %%

def sift_feature_matching_ransac(img1, img2, ratio_test=0.75, ransac_thresh=5.0, output_name='pair1_2.jpg'):
    # Step 1: Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Step 2: Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Step 3: Use BFMatcher to match descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Step 4: Find the top two matches for each descriptor in img1
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Step 5: Apply ratio test to keep only good matches
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_test * n.distance:
            good_matches.append(m)

    # Step 6: Extract matched keypoints positions
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    # Step 7: RANSAC to filter out outliers 
    best_H, inliers, outliers = ransac_outlier_rejection(pts1, pts2, ransac_thresh)
    # Step 8: Plot inliers in blue and outliers in red
    plot_inliers_outliers(img1, img2, pts1, pts2, inliers, outliers, output_name)

    H_inliner, inlier_pts1, inlier_pts2= compute_homography_from_inliers(pts1, pts2, inliers)

    # H_refined = refine_homography(H_inliner, inlier_pts1, inlier_pts2)
    
    # Extra credit
    initial_error = compute_average_error(H_inliner, inlier_pts1, inlier_pts2)
    H_refined=refine_homography_lm(H_inliner, inlier_pts1, inlier_pts2, max_iters=100, lambda_init=1e-3, tol=1e-6)
    refined_error = compute_average_error(H_refined, inlier_pts1, inlier_pts2)
    # Improvement percentage
    improvement = 100 * (initial_error - refined_error) / initial_error
    print(f"Improvement after LM refinement: {improvement:.2f}%")

    return H_refined




# %%
def warp_image(img, H, output_shape):
    """
    Warp the input image 'img' using the homography 'H' into the 'output_shape' frame.
    Apply scaling and interpolation.
    """
    warped_img = cv2.warpPerspective(img, H, output_shape, flags=cv2.INTER_LINEAR)
    return warped_img

def compute_output_size_and_offset(images, homographies):
    """
    Compute the size of the output panorama and the necessary translation to ensure all images fit.
    images: List of input images
    homographies: List of homographies relative to the reference frame
    """
    # Initialize variables to keep track of the overall bounding box
    min_x, min_y = 0, 0
    max_x, max_y = images[0].shape[1], images[0].shape[0]

    # Transform the corners of each image and track the bounding box
    for i, H in enumerate(homographies):
        h, w = images[i].shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        projected_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)

        # Update bounding box limits
        min_x = min(min_x, np.min(projected_corners[:, 0]))
        min_y = min(min_y, np.min(projected_corners[:, 1]))
        max_x = max(max_x, np.max(projected_corners[:, 0]))
        max_y = max(max_y, np.max(projected_corners[:, 1]))

    # Compute the overall output size
    output_shape = (int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y)))

    # Compute the translation homography to shift the mosaic to positive coordinates
    offset_x = -min_x if min_x < 0 else 0
    offset_y = -min_y if min_y < 0 else 0
    translation_H = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])

    return output_shape, translation_H

def create_mosaic(images, pairwise_homographies):
    """
    Create a mosaic by projecting all images onto a fixed common frame using pairwise homographies.
    pairwise_homographies: Homographies between consecutive images.
    """
    # Assume the middle image is the reference frame
    mid_idx = len(images) // 2
    H_to_ref_frame = [np.eye(3) for _ in range(len(images))]  # Homography to the reference frame

    # Compute the homographies relative to the middle (reference) image
    for i in range(mid_idx - 1, -1, -1):  # Left of the middle image
        H_to_ref_frame[i] = np.dot(H_to_ref_frame[i + 1], pairwise_homographies[i])

    for i in range(mid_idx + 1, len(images)):  # Right of the middle image
        H_to_ref_frame[i] = np.dot(H_to_ref_frame[i - 1], np.linalg.inv(pairwise_homographies[i - 1]))

    # Compute the output size and translation to fit all images
    output_shape, translation_H = compute_output_size_and_offset(images, H_to_ref_frame)

    # Initialize the mosaic
    mosaic = np.zeros((output_shape[1], output_shape[0], 3), dtype=np.uint8)

    # Warp each image to the mosaic frame
    for i, img in enumerate(images):
        # Combine homographies to map to the mosaic frame
        H_to_mosaic = np.dot(translation_H, H_to_ref_frame[i])

        # Warp the current image
        warped_img = warp_image(img, H_to_mosaic, output_shape)

        # Blend the warped image into the mosaic
        mosaic = np.where(warped_img > 0, warped_img, mosaic)  # Simple blending by overwriting empty pixels

    return mosaic




# %%
img_1 = cv2.imread('pics/1.jpg')
img_2 = cv2.imread('pics/2.jpg')
img_3 = cv2.imread('pics/3.jpg')
img_4 = cv2.imread('pics/4.jpg')
img_5 = cv2.imread('pics/5.jpg')

H_refined_1 = sift_feature_matching_ransac(img_1, img_2, output_name='pair1_2.jpg')
H_refined_2 = sift_feature_matching_ransac(img_2, img_3, output_name='pair2_3.jpg')
H_refined_3 = sift_feature_matching_ransac(img_3, img_4, output_name='pair3_4.jpg')
H_refined_4 = sift_feature_matching_ransac(img_4, img_5, output_name='pair4_5.jpg')

# List of 5 images
images = [img_1, img_2, img_3, img_4, img_5]

# List of 4 pairwise homographies between consecutive images
homographies = [H_refined_1, H_refined_2, H_refined_3, H_refined_4]

# Create the mosaic
mosaic = create_mosaic(images, homographies)

# Save the mosaic image
cv2.imwrite('mosaic_output.png', mosaic)


# img_6 = cv2.imread('IMG_9722.jpg')
# img_7 = cv2.imread('IMG_9723.jpg')
# img_8 = cv2.imread('IMG_9724.jpg')
# img_9 = cv2.imread('IMG_9725.jpg')
# img_10 = cv2.imread('IMG_9726.jpg')

# H_refined_5 = sift_feature_matching_ransac(img_6, img_7, output_name='pair6_7.jpg')
# H_refined_6 = sift_feature_matching_ransac(img_7, img_8, output_name='pair7_8.jpg')
# H_refined_7 = sift_feature_matching_ransac(img_8, img_9, output_name='pair8_9.jpg')
# H_refined_8 = sift_feature_matching_ransac(img_9, img_10, output_name='pair9_10.jpg')


# # List of 5 images
# images = [img_6, img_7, img_8, img_9, img_10]

# # List of 4 pairwise homographies between consecutive images
# homographies = [H_refined_5, H_refined_6, H_refined_7, H_refined_8]

# # Create the mosaic
# mosaic = create_mosaic(images, homographies)

# # Save the mosaic image
# cv2.imwrite('mosaic_output_custom.png', mosaic)



