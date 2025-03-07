import cv2
import numpy as np
import os

# Function to preprocess the fingerprint image
def preprocess_fingerprint(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Error: Image not found at {image_path}")
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

    if image is None:
        raise FileNotFoundError(f"❌ Error: Image not found at {image_path}")

    image = cv2.resize(image, (128, 128))  # Resize for consistency
    blurred = cv2.GaussianBlur(image, (5, 5), 0)  # Reduce noise
    edges = cv2.Canny(blurred, 50, 150)  # Edge detection
    return edges

# Function to extract fingerprint features using ORB
def extract_features(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Error: Image not found at {image_path}")
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"❌ Error: Image not found at {image_path}")

    orb = cv2.ORB_create()  # ORB feature extractor
    keypoints, descriptors = orb.detectAndCompute(image, None)  # Get keypoints & descriptors

    if descriptors is None:
        raise ValueError(f"⚠️ No features detected in {image_path}")

    return descriptors

# Example usage
if __name__ == "__main__":
    # Update path according to your specified location
    image_path = r"C:\Users\vundr\OneDrive\Desktop\Fingerprint_ML\dataset\fingerprint\sample_fingerprint.png"
    preprocessed_dir = r"C:\Users\vundr\OneDrive\Desktop\Fingerprint_ML\dataset\preprocessed"
    features_dir = r"C:\Users\vundr\OneDrive\Desktop\Fingerprint_ML\dataset\features"

    # Ensure directories exist, create if they don't
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    # Set file paths
    preprocessed_path = os.path.join(preprocessed_dir, "sample_fingerprint.png")
    features_path = os.path.join(features_dir, "sample_fingerprint.npy")

    # Debugging: print the paths
    print(f"Attempting to load image from: {image_path}")
    
    try:
        # Check if original image exists before preprocessing
        if not os.path.exists(image_path):
            print(f"❌ Image not found at: {image_path}")
        
        processed_image = preprocess_fingerprint(image_path)
        cv2.imwrite(preprocessed_path, processed_image)  # Save preprocessed image

        # Check if preprocessed image exists before extracting features
        if not os.path.exists(preprocessed_path):
            print(f"❌ Preprocessed image not found at: {preprocessed_path}")
        
        features = extract_features(preprocessed_path)
        np.save(features_path, features)  # Save features

        print("✅ Feature shape:", features.shape)
    except Exception as e:
        print(str(e))
