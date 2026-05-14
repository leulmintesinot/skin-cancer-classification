import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

# Image size (must match data.py)
IMG_HEIGHT = 224
IMG_WIDTH = 224

def build_model(num_classes, base_trainable=False, learning_rate=1e-4):
    """
    Build a CNN model for skin lesion classification using Transfer Learning.

    Args:
        num_classes (int): Number of classes (7 for HAM10000)
        base_trainable (bool): Whether to train the base model
        learning_rate (float): Learning rate for optimizer

    Returns:
        model: compiled Keras model
    """

    # Load pre-trained EfficientNetB0 without top layers
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    base_model.trainable = base_trainable  # freeze base by default

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Test run
if __name__ == "__main__":
    model = build_model(num_classes=7)
    model.summary()