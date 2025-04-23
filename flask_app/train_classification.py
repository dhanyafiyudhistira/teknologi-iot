# train_classification.py

import os
import glob
import random
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ------------------- CONFIG -------------------
# Path dataset kamu
DATASET_PATH = 'C:/Users/ACER/Downloads/rdd2020-dataset/train/Japan'
ANNOTATIONS_DIR = os.path.join(DATASET_PATH, 'annotations')
IMAGES_DIR = os.path.join(DATASET_PATH, 'images')

# Parameter
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30  # Increased max epochs since we have early stopping
SAVE_MODEL_PATH = 'saved_models/classifier_model.keras'  # Changed to .keras extension
CHECKPOINT_PATH = 'saved_models/checkpoint_model.keras'  # Changed to .keras extension

# ------------------- PERSIAPAN DATASET -------------------

# Cari nama file yang ada bounding box (rusak) dari file XML
import xml.etree.ElementTree as ET

def get_images_with_damage(annotations_dir):
    damaged_images = set()
    for xml_file in glob.glob(os.path.join(annotations_dir, '*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.findall('object'):  # kalau ada object berarti rusak
            filename = root.find('filename').text
            damaged_images.add(filename)
    return damaged_images

print("Mengambil daftar gambar rusak...")
damaged_images = get_images_with_damage(ANNOTATIONS_DIR)

# Semua gambar
all_images = [os.path.basename(p) for p in glob.glob(os.path.join(IMAGES_DIR, '*.jpg'))]

# Klasifikasikan gambar rusak/tidak rusak
image_labels = []
for img_name in all_images:
    label = 'damaged' if img_name in damaged_images else 'normal'
    image_labels.append((img_name, label))

# Cek distribusi kelas
damaged_count = sum(1 for _, label in image_labels if label == 'damaged')
normal_count = sum(1 for _, label in image_labels if label == 'normal')
print(f"Jumlah gambar rusak: {damaged_count}, normal: {normal_count}")

# Acak dataset
random.shuffle(image_labels)

# Split dataset
split_idx = int(0.8 * len(image_labels))
train_data = image_labels[:split_idx]
val_data = image_labels[split_idx:]

# ------------------- DATA GENERATOR -------------------

# Buat folder sementara untuk ImageDataGenerator
if not os.path.exists('dataset_temp'):
    os.makedirs('dataset_temp/train/damaged')
    os.makedirs('dataset_temp/train/normal')
    os.makedirs('dataset_temp/val/damaged')
    os.makedirs('dataset_temp/val/normal')

# Copy file ke folder sementara
for img_name, label in train_data:
    shutil.copy(os.path.join(IMAGES_DIR, img_name), f'dataset_temp/train/{label}/{img_name}')
for img_name, label in val_data:
    shutil.copy(os.path.join(IMAGES_DIR, img_name), f'dataset_temp/val/{label}/{img_name}')

# Data Generator dengan augmentasi untuk mengatasi imbalance dan mencegah overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset_temp/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    'dataset_temp/val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ------------------- MODEL -------------------
print("Membuat model CNN...")

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),  # Tambahan layer untuk meningkatkan kapasitas model
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
    loss='binary_crossentropy', 
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# ------------------- CALLBACKS -------------------

# Buat folder untuk menyimpan model jika belum ada
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# Early stopping untuk mencegah overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',  # Memantau validation loss
    patience=10,         # Berhenti jika tidak ada peningkatan selama 10 epochs
    verbose=1,
    restore_best_weights=True  # Mengembalikan model ke bobot terbaik
)

# Model checkpoint untuk menyimpan model terbaik
checkpoint = ModelCheckpoint(
    CHECKPOINT_PATH,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)

# ------------------- TRAINING -------------------
print("Mulai training...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping, checkpoint],  # Menggunakan callback
    class_weight={  # Mengatasi ketidakseimbangan kelas jika perlu
        0: 1.0,  # normal
        1: damaged_count / normal_count if normal_count > damaged_count else 1.0  # damaged
    }
)

# ------------------- SAVE MODEL -------------------
model.save(SAVE_MODEL_PATH)
print(f"Model saved to {SAVE_MODEL_PATH}")

# ------------------- EVALUASI MODEL -------------------
print("\nEvaluasi Model:")
val_loss, val_acc, val_precision, val_recall = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")

# ------------------- PLOT TRAINING HISTORY -------------------
import matplotlib.pyplot as plt

# Plot training & validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig('training_history.png')
print("Training history plot saved as 'training_history.png'")

# ------------------- BERSIHIN FOLDER TEMP -------------------
shutil.rmtree('dataset_temp')
print("Folder sementara dihapus, training selesai!")