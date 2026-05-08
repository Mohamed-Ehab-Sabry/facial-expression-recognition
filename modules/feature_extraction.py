import skimage
import sklearn
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Point to the physical model file you downloaded
base_options = python.BaseOptions(model_asset_path='../modules/models/blaze_face_short_range.tflite')

# 2. Configure the options (IMAGE mode is perfect for your FER dataset)
options = vision.FaceDetectorOptions(base_options=base_options)

# 3. Create the detector instance
detector = vision.FaceDetector.create_from_options(options)

def extract_face_roi(preprocessed_img, target_size=(128, 128), upsample_size=(200, 200), margin=0.1, fallback=True):
    """
    Extract face region from a preprocessed grayscale image.
    
    Args:
        preprocessed_img: 2D np.uint8 grayscale image (48x48 from FER2013)
        target_size: desired output size (width, height) for the cropped face
        upsample_size: size to upscale before detection (width, height)
        margin: extra margin around detected face box (as fraction of box dimensions)
        fallback: if True, return resized original image when no face is detected
    
    Returns:
        Cropped and resized face image (grayscale) or resized original if fallback.
    """
    h, w = preprocessed_img.shape
    us_w, us_h = upsample_size

    # 1. Upsample to give the model more detail
    img_upsampled = cv2.resize(preprocessed_img, upsample_size, interpolation=cv2.INTER_CUBIC)
    
    # 2. CRITICAL: Convert to MediaPipe Image (requires RGB)
    img_rgb = cv2.cvtColor(img_upsampled, cv2.COLOR_GRAY2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    # 3. Detect faces
    result = detector.detect(mp_image)
    
    # 4. Process results
    if result.detections:
        # Pick the first detected face
        detection = result.detections[0]
        
        # MediaPipe returns normalized coordinates (0.0 to 1.0)
        bbox = detection.bounding_box
        xmin = int(bbox.origin_x * us_w)
        ymin = int(bbox.origin_y * us_h)
        box_w = int(bbox.width * us_w)
        box_h = int(bbox.height * us_h)

        # Add margin
        margin_w = int(box_w * margin)
        margin_h = int(box_h * margin)
        x1 = max(0, xmin - margin_w)
        y1 = max(0, ymin - margin_h)
        x2 = min(us_w, xmin + box_w + margin_w)
        y2 = min(us_h, ymin + box_h + margin_h)

        
        # --- Safety check: ensure crop region is non-empty ---
        if x1 >= x2 or y1 >= y2:
            # Invalid bounding box → use fallback
            if fallback:
                return cv2.resize(preprocessed_img, target_size, interpolation=cv2.INTER_LINEAR)
            else:
                return None

        # Crop and resize
        face_crop = img_upsampled[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_LINEAR)
        return face_resized
    
    # Fallback: no face detected
    if fallback:
        return cv2.resize(preprocessed_img, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        return None

def extract_hog_features():
    pass

def extract_lpb_features():
    pass

def extract_edge_features():
    pass

def combine_features():
    pass

