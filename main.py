import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Parameters
IMG_SIZE = 64  # Resize all images to 64x64
DATA_DIR = 'data'
CATEGORIES = ['cats', 'dogs']

def load_data():
    features = []
    labels = []
    for label, category in enumerate(CATEGORIES):
        folder = os.path.join(DATA_DIR, category)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue  # skip unreadable files
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale for SVM
            features.append(img.flatten())  # Flatten 2D image to 1D
            labels.append(label)
    return np.array(features), np.array(labels)

# Load data
print("[INFO] Loading data...")
X, y = load_data()
print(f"[INFO] Loaded {len(X)} samples.")

# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
print("[INFO] Training SVM...")
clf = SVC(kernel='linear')  # You can also try 'rbf' or 'poly'
clf.fit(X_train, y_train)

# Predict
print("[INFO] Evaluating model...")
y_pred = clf.predict(X_test)

# Report
print(classification_report(y_test, y_pred, target_names=CATEGORIES))

# Optional: visualize a few test predictions
def show_samples():
    for i in range(5):
        img = X_test[i].reshape(IMG_SIZE, IMG_SIZE)
        plt.imshow(img, cmap='gray')
        plt.title(f"Predicted: {CATEGORIES[y_pred[i]]}")
        plt.axis('off')
        plt.show()
        # Predict on the test set
y_pred = clf.predict(X_test)

# Show results
from sklearn.metrics import classification_report, confusion_matrix
print("[INFO] Classification Report:")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))
import matplotlib.pyplot as plt

def show_sample_predictions():
    for i in range(5):
        img = X_test[i].reshape(IMG_SIZE, IMG_SIZE)
        plt.imshow(img, cmap='gray')
        plt.title(f"Predicted: {CATEGORIES[y_pred[i]]}, Actual: {CATEGORIES[y_test[i]]}")
        plt.axis('off')
        plt.show()

# Call the function (optional)
show_sample_predictions()



# Uncomment to view test results
# show_samples()
