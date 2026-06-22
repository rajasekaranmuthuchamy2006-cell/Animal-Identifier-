import tensorflow as tf
from tensorflow.keras import layers, models

# Load data directly from the dataset folder structural layout
X_train = tf.keras.utils.image_dataset_from_directory(
    'dataset',
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224),
    batch_size=32
)

# Build a fast, lightweight CNN
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(X_train.class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("🚀 Starting training...")
model.fit(X_train, epochs=15) # Increase epochs to 25 or 30 for better accuracy!

# Save your model files cleanly
model.save('animal_model.keras')
print("✅ Saved completely as 'animal_model.keras'!")