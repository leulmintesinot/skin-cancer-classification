import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# Paths to your image folders and metadata
IMG_DIRS = [
    "./data/HAM10000_images_part_1",
    "./data/HAM10000_images_part_2"
]
METADATA_PATH = "./data/HAM10000_metadata.csv"

# Image size for CNN input
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Load metadata
def load_metadata():
    df = pd.read_csv(METADATA_PATH)
    return df

# Map image_id to full image path
def get_image_path(image_id):
    for dir_path in IMG_DIRS:
        path = os.path.join(dir_path, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None

# Prepare dataframe with full paths
def prepare_dataframe():
    df = load_metadata()
    df["path"] = df["image_id"].apply(get_image_path)
    df = df.dropna(subset=["path"])  # remove missing files
    return df

# Split data into train, validation, test
def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["dx"], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df["dx"], random_state=42)
    return train_df, val_df, test_df

# Create Keras image generators
def create_generators(train_df, val_df, test_df):
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,        #skin lesions can be flipped any way
        zoom_range=0.2,             
        shear_range=0.1,            
        brightness_range=[0.8, 1.2] #handles lighting variation
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col="path",
        y_col="dx",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_generator = val_test_datagen.flow_from_dataframe(
        val_df,
        x_col="path",
        y_col="dx",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_dataframe(
        test_df,
        x_col="path",
        y_col="dx",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_generator, val_generator, test_generator

# Main function to load everything
def load_data():
    df = prepare_dataframe()
    train_df, val_df, test_df = split_data(df)
    train_gen, val_gen, test_gen = create_generators(train_df, val_df, test_df)
    num_classes = len(df["dx"].unique())
    return train_gen, val_gen, test_gen, num_classes

def compute_weights(train_df):
    """Compute class weights to counteract imbalance."""
    classes = train_df["dx"].unique()
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.sort(classes),  
        y=train_df["dx"]
    )
    # Map to integer indices (Keras expects {0: w0, 1: w1, ...})
    label_to_index = {label: i for i, label in enumerate(sorted(classes))}
    class_weight_dict = {label_to_index[cls]: w for cls, w in zip(np.sort(classes), class_weights_array)}
    return class_weight_dict

def load_data():
    df = prepare_dataframe()
    train_df, val_df, test_df = split_data(df)
    train_gen, val_gen, test_gen = create_generators(train_df, val_df, test_df)
    num_classes = len(df["dx"].unique())
    class_weights = compute_weights(train_df)   # <-- NEW
    return train_gen, val_gen, test_gen, num_classes, class_weights  # <-- add class_weights

# Test run
if __name__ == "__main__":
    train_gen, val_gen, test_gen, num_classes = load_data()
    print("Number of classes:", num_classes)
    print("Training batches:", len(train_gen))