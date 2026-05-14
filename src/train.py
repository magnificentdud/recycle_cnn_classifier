# placeholder for training script

"""
train.py
Training script for the recycle classification model.
"""

import os
from dataset import create_generators
from model import build_model
import tensorflow as tf

def train_model(
    data_dir,
    img_size=(224, 224),
    batch_size=32,
    epochs=10,
    save_path="saved_models/recycle_mobilenetv2.h5"
):
    """
    Trains the MobileNetV2 model using transfer learning.

    Args:
        data_dir (str): Path to dataset root.
        img_size (tuple): Image size.
        batch_size (int): Batch size.
        epochs (int): Number of epochs.
        save_path (str): Where to save the trained model.
    """

    train_gen, val_gen, _ = create_generators(data_dir, img_size, batch_size)

    num_classes = train_gen.num_classes
    model = build_model(input_shape=(*img_size, 3), num_classes=num_classes)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)

    return history, model

if __name__ == "__main__":
    DATA_DIR = "data"  # adjust if needed
    train_model(DATA_DIR)
