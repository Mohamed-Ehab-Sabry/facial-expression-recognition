import sklearn
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from skimage.feature import hog, local_binary_pattern

model_path='../modules/models/blaze_face_short_range.tflite'

try:
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)
    print("MediaPipe Face Detector initialized successfully!")
except Exception as e:
    print(f"Error initializing detector. Did you download the .tflite model?\n{e}")

def extract_face_roi(image_path, target_size=(48, 48)):
    """
    Extracts, crops, and resizes a face Region of Interest (ROI) from an image.

    This function reads a grayscale image from the given path, temporarily converts 
    it to RGB to fulfill MediaPipe's input requirements, and uses a pre-initialized 
    MediaPipe Face Detector to locate the bounding box of the primary face. It then 
    crops this bounding box from the original grayscale image and resizes it to a 
    standardized dimension for feature extraction. 

    If no face is detected by MediaPipe, or if the bounding box results in an empty 
    crop, the function safely falls back to returning the resized original image 
    to prevent dataset processing loops from crashing.

    Args:
        image_path (str): The file path to the input grayscale image.
        target_size (tuple of int, optional): The desired output dimensions 
            (width, height) for the returned face image. Defaults to (48, 48).

    Returns:
        numpy.ndarray: A 2D NumPy array representing the cropped, resized 
            grayscale face. Returns None if the image file cannot be read.
    """
    # 1. Read the preprocessed image in Grayscale
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if gray_img is None:
        print(f"Could not read image at {image_path}")
        return None
        
    img_height, img_width = gray_img.shape

    # 2. Convert to RGB purely for MediaPipe detection
    rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    
    # 3. Convert to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    
    # 4. Run detection
    detection_result = detector.detect(mp_image)
    
    # 5. Handle cases where no face is found
    if not detection_result.detections:
        # Fallback: Just resize the whole original image if MediaPipe fails
        # This prevents the training loop from crashing later.
        print('=' * 20)
        print('no detection')
        print('=' * 20)
        return cv2.resize(gray_img, target_size)
        
    # 6. Extract bounding box of the FIRST face detected
    bbox = detection_result.detections[0].bounding_box
    
    # Extract coordinates (MediaPipe Tasks API gives absolute pixels)
    x = bbox.origin_x
    y = bbox.origin_y
    w = bbox.width
    h = bbox.height
    
    # 7. Safety check: Ensure coordinates don't go outside the image boundaries
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(img_width, x + w)
    y_end = min(img_height, y + h)
    
    # 8. Crop from the ORIGINAL GRAYSCALE image
    cropped_face = gray_img[y_start:y_end, x_start:x_end]
    
    # 9. Safety check: If the crop is somehow empty, return fallback
    if cropped_face.size == 0:
        return cv2.resize(gray_img, target_size)
        
    # 10. Resize to standard shape for HOG/LBP extraction later
    final_face = cv2.resize(cropped_face, target_size)
    
    return final_face

def extract_hog_features(img, pixels_per_cell = (8, 8), cells_per_block = (2, 2), orientations = 9, visualize = False):
    """
    Extract HOG features from a face image.
    
    Args:
        img: 2D grayscale image (e.g., 48x48)
        pixels_per_cell: cell size for gradient histogram
        cells_per_block: block size for normalisation
        orientations: number of orientation bins
    
    Returns:
        1D numpy array of HOG features
    """
    
    if visualize:
        # Visualize HOG (optional for report/debugging)
        _, hog_image = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block, visualize=True)
        
        plt.figure(figsize= (8, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap= 'gray')
        plt.title('original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(hog_image, cmap= 'gray')
        plt.title('HOG output')
        plt.axis('off')
        plt.show()
    
    features = hog(
        img, orientations = orientations,
        pixels_per_cell= pixels_per_cell,
        cells_per_block = cells_per_block,
        block_norm='L2-Hys', transform_sqrt=True,
        feature_vector=True
    )
    
    return features

def extract_lbp_features(img, n_points = 8, radius = 1, method = 'uniform', visualize = False):
    """
    Extracts Local Binary Pattern (LBP) texture features from an image as a normalized histogram.

    This function computes the LBP representation of the input image to capture 
    micro-textures. It then calculates a frequency histogram of these texture 
    patterns and normalizes it. Using the 'uniform' method is highly recommended 
    as it drastically reduces the number of bins (preventing the \"curse of 
    dimensionality\") and makes the model less prone to overfitting.

    Args:
        img (numpy.ndarray): The 2D cropped, grayscale face ROI.
        n_points (int, optional): Number of circularly symmetric neighbor points. Defaults to 8.
        radius (int, optional): Radius of the circle for the neighbors. Defaults to 1.
        method (str, optional): Method to determine the pattern. Defaults to 'uniform'.
        visualize (bool, optional): If True, plots the original image alongside 
            its 2D LBP representation for debugging. Defaults to False.

    Returns:
        numpy.ndarray: A 1D array representing the normalized histogram of LBP codes.
    """      
    
    # method = 'uniform' It drastically reduces the number of patterns and makes the model less prone to overfitting.
    
    lbp = local_binary_pattern(img, P= n_points, R= radius, method= method)
    
    # Number of bins for 'uniform' pattern with n_points = n_points+2
    n_bins = n_points + 2 if method == 'uniform' else 2**n_points
    
    (hist, _) = np.histogram(lbp.ravel(), bins= n_bins, range= (0, n_bins))
    
    hist = hist.astype('float')
    # Normalize histogram
    hist /= (hist.sum() + 1e-7)
    
    if visualize:
        plt.figure(figsize= (8, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap= 'gray')
        plt.title('original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(lbp, cmap= 'gray')
        plt.title('LBP Output')
        plt.axis('off')
        plt.show()
    
    return hist

def extract_edge_features(img, low_threshold = 50, high_threshold = 200, visualize = False):
    """
    Extracts outline features by computing an orientation histogram of Canny edge pixels.

    This function isolates sharp facial transitions (like the outline of lips or eyes) 
    using the Canny edge detector. To translate these edges into a machine-learning 
    friendly format, it calculates the gradient orientation (0-180 degrees) of the 
    image using Sobel filters, filters for only the pixels that are edges, and 
    buckets them into an 18-bin normalized histogram (10 degrees per bin).

    Args:
        img (numpy.ndarray): The 2D cropped, grayscale face ROI.
        low_threshold (int, optional): First threshold for the Canny hysteresis procedure. Defaults to 50.
        high_threshold (int, optional): Second threshold for the Canny hysteresis procedure. Defaults to 200.

    Returns:
        numpy.ndarray: A 1D array of length 18 representing the normalized edge orientation histogram.
    """
    
    edges = cv2.Canny(img, low_threshold, high_threshold)
    
    # Compute gradient orientation of the original image, but only at edge pixels (using sobel)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize= 3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize= 3)
    
    # Compute orientation in degrees (0-180)
    orientation = np.arctan2(sobely, sobelx) * 180 / np.pi % 180
    
    # Select only edge pixels (where edges > 0)
    edge_orientations = orientation[edges > 0]
    
    if visualize:
        plt.Figure(figsize= (8, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap= 'gray')
        plt.title('original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(edges, cmap= 'gray')
        plt.title('LBP Output')
        plt.axis('off')
        plt.show()
    
    # Histogram with 18 bins (10° each)
    hist, _ = np.histogram(edge_orientations, bins=18, range=(0, 180))
    hist = hist.astype('float')
    
    # Normalize
    hist /= (hist.sum() + 1e-7)
    
    return hist

def extract_all_features(img):
    """
    Combines shape, texture, and edge features into a single master feature vector.

    This function calls the individual feature extractors (HOG, LBP, and Canny Edge 
    Orientations) on a single cropped face ROI. It concatenates their 1D outputs 
    end-to-end to create a dense, comprehensive representation of the facial expression 
    suitable for classic machine learning classifiers (SVM, Random Forest, etc.).

    Args:
        img (numpy.ndarray): The 2D cropped, grayscale face ROI.

    Returns:
        numpy.ndarray: A single, flat 1D array containing the concatenated features.
    """
    
    hog_features = extract_hog_features(img)
    lbp_features = extract_lbp_features(img)
    canny_edge_features = extract_edge_features(img)
    
    combined = np.concatenate([hog_features, lbp_features, canny_edge_features])
    
    return combined
