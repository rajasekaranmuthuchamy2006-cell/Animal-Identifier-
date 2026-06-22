import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

DATASET_DIR = "dataset"
TARGET_SIZE = (224, 224)

def prepare_dataset():
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: '{DATASET_DIR}' folder not found! Create it and add animal folders.")
        return

    classes = sorted(os.listdir(DATASET_DIR))
    print(f"📦 Found classes: {classes}")
    
    # Save the label names to animals.txt automatically
    with open("animals.txt", "w") as f:
        for c in classes:
            f.write(f"{c}\n")
    print("📝 Saved classes to animals.txt")

if __name__ == "__main__":
    prepare_dataset()