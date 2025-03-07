import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import pickle
from datetime import datetime
import pandas as pd
import os

# Load the trained model
with open("fingerprint_model.pkl", "rb") as f:
    model = pickle.load(f)

# Attendance file path
attendance_file = "attendance.csv"

# Function to preprocess the fingerprint image
def preprocess_fingerprint(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.resize(image, (128, 128))  # Resize to match training size
    blurred = cv2.GaussianBlur(image, (5, 5), 0)  # Noise reduction
    edges = cv2.Canny(blurred, 50, 150)  # Edge detection
    return edges

# Function to extract ORB features
def extract_features(image_path):
    image = preprocess_fingerprint(image_path)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)

    if descriptors is None:
        raise ValueError(f"No features detected in {image_path}")

    return descriptors

# Function to recognize the fingerprint
def recognize_fingerprint(image_path):
    descriptors = extract_features(image_path)
    descriptors = descriptors.flatten().reshape(1, -1)  # Flatten the descriptors
    user_id = model.predict(descriptors)
    return user_id[0]

# Function to mark attendance
def mark_attendance(user_id):
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
    else:
        df = pd.DataFrame(columns=["User ID", "Name", "Time"])

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_name = f"User_{user_id}"
    df = df.append({"User ID": user_id, "Name": user_name, "Time": current_time}, ignore_index=True)
    df.to_csv(attendance_file, index=False)
    return f"Attendance marked for User {user_id} at {current_time}"

# Function to handle fingerprint recognition
def on_recognize_button_click():
    try:
        image_path = "C:/path_to_fingerprint_image/sample_fingerprint.png"  # Update with your image path
        user_id = recognize_fingerprint(image_path)
        result_message = f"User ID: {user_id} recognized successfully!"
        attendance_message = mark_attendance(user_id)
        result_label.config(text=result_message)
        attendance_label.config(text=attendance_message)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the Tkinter window
window = tk.Tk()
window.title("Fingerprint Attendance System")
window.geometry("500x400")

# Create and place widgets
title_label = tk.Label(window, text="Fingerprint Attendance System", font=("Arial", 16), pady=20)
title_label.pack()

instruction_label = tk.Label(window, text="Please place your finger on the scanner", font=("Arial", 12))
instruction_label.pack(pady=10)

recognize_button = tk.Button(window, text="Recognize Fingerprint", font=("Arial", 12), command=on_recognize_button_click)
recognize_button.pack(pady=10)

result_label = tk.Label(window, text="Recognition Result: ", font=("Arial", 12))
result_label.pack(pady=10)

attendance_label = tk.Label(window, text="Attendance Status: ", font=("Arial", 12))
attendance_label.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()
