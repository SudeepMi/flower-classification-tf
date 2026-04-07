import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pathlib

# Load dataset
img_size = (180, 180)
batch_size = 32


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = pathlib.Path(
    tf.keras.utils.get_file(
        fname="flower_photos",
        origin=dataset_url,
        untar=True
    )
)
if not data_dir.exists() or not data_dir.is_dir():
    raise FileNotFoundError(f"Could not locate extracted dataset directory: {data_dir}")

# On some setups, extraction produces: .../flower_photos/flower_photos/<class_dirs>
nested_dir = data_dir / "flower_photos"
if nested_dir.exists() and nested_dir.is_dir():
    data_dir = nested_dir

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
if len(class_names) < 2:
    raise ValueError(f"Expected multiple classes, found: {class_names}")

# Normalize
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(180,180,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# Save model
model.save("flower_model.h5")