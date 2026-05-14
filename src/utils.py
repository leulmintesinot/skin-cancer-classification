import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# -----------------------------
# 1️⃣ Evaluate model & save report
# -----------------------------
def evaluate_model(model, test_gen, class_names, save_path="./results/classification_report.txt"):
    y_pred = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes

    # Accuracy
    loss, acc = model.evaluate(test_gen, verbose=0)
    print(f"Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")

    # Classification report
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    print("Classification Report:\n", report)

    # Save report
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(report)
    print(f"Classification report saved at: {save_path}")

    return loss, acc, report

# -----------------------------
# 2️⃣ Confusion Matrix
# -----------------------------
def plot_confusion_matrix(model, test_gen, class_names, save_path="./results/confusion_matrix.png"):
    y_pred = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes

    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
    print(f"Confusion matrix saved at: {save_path}")

# -----------------------------
# 3️⃣ Grad-CAM
# -----------------------------
def grad_cam(model, img_path, last_conv_layer_name="top_conv", size=(224, 224), save_path=None):
    img = image.load_img(img_path, target_size=size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        pred_index = np.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    orig_img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)

    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Grad-CAM")
    plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, superimposed_img)
        print(f"Grad-CAM saved at: {save_path}")

# -----------------------------
# 4️⃣ Batch Grad-CAM
# -----------------------------
def grad_cam_batch(model, img_paths, save_dir="./results/gradcam/", last_conv_layer_name="top_conv", size=(224, 224)):
    os.makedirs(save_dir, exist_ok=True)
    for img_path in img_paths:
        img_name = os.path.basename(img_path).replace(".jpg", "_heatmap.png")
        save_path = os.path.join(save_dir, img_name)
        grad_cam(model, img_path, last_conv_layer_name=last_conv_layer_name, size=size, save_path=save_path)