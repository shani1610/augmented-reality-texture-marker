# Import necessary libraries
import cv2
import numpy as np

# Display the versions of used packages
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)

# Function to perform feature matching
def feature_matching(img1, img2, ransac_reproj_threshold=0.0):

    # # Initialize the ORB detector
    # orb = cv2.ORB_create()

    # # Find the keypoints and descriptors with ORB
    # kp1, des1 = orb.detectAndCompute(img1, None)
    # kp2, des2 = orb.detectAndCompute(img2, None)

    # Initialize the ORB detector
    orb = cv2.xfeatures2d.SURF_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Initialize the Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors using BFMatcher
    matches = bf.match(des1, des2)

    # Sort them based on distance
    matches = sorted(matches, key=lambda x: x.distance)

    if ransac_reproj_threshold!=0.0:
        # Use RANSAC to filter outliers
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find the fundamental matrix using RANSAC
        fundamental_matrix, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, ransac_reproj_threshold)

        # Apply the mask to keep only the inliers
        inliers = mask.ravel() == 1
        matches = [matches[i] for i in range(len(matches)) if inliers[i]]

    # Draw the inlier matches on the images
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the result
    cv2.imshow('Feature Matching', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Paths to the input images
image1_path = './data/box.jpeg'
image2_path = './data/box_scene1.jpeg'

# Read the input images
img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

# Define the percentage by which you want to resize (e.g., 50%)
resize_percentage = 40

# Calculate the new width and height
new_width1 = int(img1.shape[1] * (resize_percentage / 100))
new_height1 = int(img1.shape[0] * (resize_percentage / 100))

new_width2 = int(img2.shape[1] * (resize_percentage / 100))
new_height2 = int(img2.shape[0] * (resize_percentage / 100))

# Resize the images to a reasonable size
resized_img1 = cv2.resize(img1, (new_width1, new_height1))
resized_img2 = cv2.resize(img2, (new_width2, new_height2))

# Perform feature matching
#feature_matching(resized_img1, resized_img2)

feature_matching(resized_img1, resized_img2, 10.0)
