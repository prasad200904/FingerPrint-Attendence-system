import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import os

# Load trained model
with open("fingerprint_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load attendance file
attendance_file = "attendance.csv"

# Function to preprocess the image (same as in training)
def preprocess_fingerprint(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if image is None:
        raise FileNotFoundError(f"❌ Error: Image not found at {image_path}")
    image = cv2.resize(image, (128, 128))  # Resize image to match training size
    blurred = cv2.GaussianBlur(image, (5, 5), 0)  # Reduce noise
    edges = cv2.Canny(blurred, 50, 150)  # Edge detection
    return edges

# Function to extract ORB features (same as in training)
def extract_features(image_path):
    image = preprocess_fingerprint(image_path)
    orb = cv2.ORB_create()  # ORB feature extractor
    keypoints, descriptors = orb.detectAndCompute(image, None)

    if descriptors is None:
        raise ValueError(f"⚠️ No features detected in {image_path}")

    return descriptors

# Function to recognize fingerprint
def recognize_fingerprint(image_path):
    descriptors = extract_features(image_path)

    # Flatten the descriptors to match the shape of the training data
    descriptors = descriptors.flatten().reshape(1, -1)

    # Dynamically adjust n_neighbors based on number of samples
    n_samples = len(model._fit_X)  # Get the number of samples in the model
    n_neighbors = min(3, n_samples)  # Use 3 or the number of samples

    # Create a new KNN model with the appropriate number of neighbors
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Re-train the model with the current data to apply new n_neighbors
    knn.fit(model._fit_X, model._y)

    # Predict the user ID
    user_id = knn.predict(descriptors)
    return user_id[0]

# Mark attendance
def mark_attendance(user_id):
    # Check if the attendance file exists
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
    else:
        df = pd.DataFrame(columns=["User ID", "Name", "Time"])

    # Get the current time and format it
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add attendance record
    user_name = f"User_{user_id}"
    df = df.append({"User ID": user_id, "Name": user_name, "Time": current_time}, ignore_index=True)

    # Save the updated attendance file
    df.to_csv(attendance_file, index=False)
    print(f"✅ Attendance marked for User {user_id} at {current_time}")

# Example usage
if __name__ == "__main__":
    # Update the path to the correct fingerprint image location
    image_path = r"C:\Users\vundr\OneDrive\Desktop\Fingerprint_ML\dataset\fingerprint\sample_fingerprint.png"  # Corrected path

    try:
        user_id = recognize_fingerprint(image_path)
        print(f"✅ Recognized User ID: {user_id}")
        mark_attendance(user_id)
    except Exception as e:
        print(str(e))
