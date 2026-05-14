import os
from tensorflow.keras.models import load_model
from data import load_data
from utils import evaluate_model, plot_confusion_matrix, grad_cam_batch

# -----------------------------
# Directories
# -----------------------------
MODEL_PATH = "./models/skin_cancer_model.h5"
RESULTS_DIR = "./results"
GRADCAM_DIR = os.path.join(RESULTS_DIR, "gradcam")
os.makedirs(GRADCAM_DIR, exist_ok=True)

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
_, val_gen, test_gen, num_classes = load_data()
class_names = list(val_gen.class_indices.keys())

# -----------------------------
# 2️⃣ Load trained model
# -----------------------------
model = load_model(MODEL_PATH)
print("Model loaded:", MODEL_PATH)

# -----------------------------
# 3️⃣ Evaluate model
# -----------------------------
loss, acc, report = evaluate_model(model, test_gen, class_names)
plot_confusion_matrix(model, test_gen, class_names)

# -----------------------------
# 4️⃣ Grad-CAM visualization
# -----------------------------
# Pick one sample per class
sample_images = []
for class_name in class_names:
    for path, label in zip(val_gen.filepaths, val_gen.labels):
        if class_names[label] == class_name:
            sample_images.append(path)
            break

grad_cam_batch(model, sample_images, save_dir=GRADCAM_DIR)
print("Grad-CAM images saved in:", GRADCAM_DIR)

# -----------------------------
# 5️⃣ Finished
# -----------------------------
print("All evaluation and visualizations completed!")