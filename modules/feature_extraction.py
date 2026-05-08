import skimage
import sklearn
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

def extract_hog_features():
    pass

def extract_lpb_features():
    pass

def extract_edge_features():
    pass

def combine_features():
    pass

