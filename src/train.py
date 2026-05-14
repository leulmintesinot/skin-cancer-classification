import os
import matplotlib.pyplot as plt

# -----------------------------
# Suppress TensorFlow GPU warnings
# -----------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data import load_data   # <-- use direct import for Colab
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Paths
MODEL_DIR = "./models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

RESULTS_DIR = "./results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 32

def train():
    # -----------------------------
    # Step 1: Load data
    # -----------------------------
    train_gen, val_gen, test_gen, num_classes, class_weights = load_data()
    print(f"Number of classes: {num_classes}")
    print(f"Class weights: {class_weights}")  
    print(f"Training batches: {len(train_gen)} | Validation batches: {len(val_gen)} | Test batches: {len(test_gen)}")

    # -----------------------------
    # Step 2: Build model
    # -----------------------------
    model = build_model(num_classes=num_classes, base_trainable=False)

    # -----------------------------
    # Step 3: Callbacks
    # -----------------------------
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "skin_cancer_model.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    callbacks = [checkpoint, early_stop, reduce_lr]

    # -----------------------------
    # Step 4: Train model
    # -----------------------------
    history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

    # -----------------------------
    # Step 5: Save training plots
    # -----------------------------
    # Accuracy plot
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_plot.png"))

    # Loss plot
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_plot.png"))

    print("Training finished! Model and plots saved.")
    return model, history, test_gen

# -----------------------------
# Run training if this script is called
# -----------------------------
if __name__ == "__main__":
    model, history, test_gen = train()