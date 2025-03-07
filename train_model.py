import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load dataset features and labels
def load_training_data(feature_dir):
    X = []
    y = []
    user_id = 0

    for file in os.listdir(feature_dir):
        if file.endswith(".npy"):
            feature_path = os.path.join(feature_dir, file)
            features = np.load(feature_path)
            X.append(features.flatten())  # Flatten feature array
            y.append(user_id)  # Assign a unique ID to each user
            user_id += 1

    return np.array(X), np.array(y)

# Load data
X_train, y_train = load_training_data("dataset/features")

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Save trained model
with open("fingerprint_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")
