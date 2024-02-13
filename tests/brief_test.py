# Import necessary libraries
import cv2
import numpy as np

# Display the versions of used packages
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)

# Function to perform feature matching using ORB
def feature_matching_orb(img1, img2, resize_percentage):
    # Calculate the new width and height for both images
    new_width = int(img1.shape[1] * (resize_percentage / 100))
    new_height = int(img1.shape[0] * (resize_percentage / 100))

    # Resize both images to the same dimensions
    resized_img1 = cv2.resize(img1, (new_width, new_height))
    resized_img2 = cv2.resize(img2, (new_width, new_height))

    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(resized_img1, None)
    kp2, des2 = orb.detectAndCompute(resized_img2, None)

    # Initialize the Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors using BFMatcher
    matches = bf.match(des1, des2)

    # Sort them in ascending order of distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw circles only around the keypoints that are part of good matches
    result = cv2.drawMatches(resized_img1, kp1, resized_img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the result
    cv2.imshow('Feature Matching Result (ORB)', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Paths to the input images
image1_path = './data/box.jpeg'
image2_path = './data/box_scene1.jpeg'

# Read the input images
img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

# Define the percentage by which you want to resize (e.g., 40%)
resize_percentage = 40

# Perform feature matching using ORB
feature_matching_orb(img1, img2, resize_percentage)
