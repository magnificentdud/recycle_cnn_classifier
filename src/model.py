# placeholder for model architecture
"""
model.py
Defines the MobileNetV2 transfer learning model for multi-class classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(224, 224, 3), num_classes=10, train_base=False):
    """
    Builds a MobileNetV2 transfer learning model.

    Args:
        input_shape (tuple): Input image shape.
        num_classes (int): Number of output classes.
        train_base (bool): Whether to unfreeze the base model for fine-tuning.

    Returns:
        tf.keras.Model
    """

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = train_base

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
