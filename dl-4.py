import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# Constants
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 2
IMAGE_CHANNELS=3
# Create ImageDataGenerator
train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True
)
test_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
TRAIN_DIR,
target_size=IMAGE_SIZE,
batch_size=BATCH_SIZE,
class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
TEST_DIR,
target_size=IMAGE_SIZE,
batch_size=BATCH_SIZE,
class_mode='binary'
)
model = Sequential([
Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, IMAGE_CHANNELS)),
BatchNormalization(),
MaxPooling2D(pool_size=(2, 2)),
#Dropout(0.25),
Conv2D(64, (3, 3), activation='relu'),
BatchNormalization(),
MaxPooling2D(pool_size=(2, 2)),
#Dropout(0.25),
( ( ) )
2/4/24, 8:15 PM DL-Experment-4.ipynb - Colaboratory
https://colab.research.google.com/drive/1BDuzWmQOruJJl8qbwEEEsPZah_XkVmaC?usp=sharing#scrollTo=87MTF-7qQ_IG&printMode=true 2/6
Conv2D(128, (3, 3), activation='relu'),
BatchNormalization(),
MaxPooling2D(pool_size=(2, 2)),
#Dropout(0.25),
Flatten(),
Dense(512, activation='relu'),
BatchNormalization(),
Dropout(0.5),
Dense(1, activation='sigmoid'), # 2 because we have cat and dog classes
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Train the model
history = model.fit(train_generator, steps_per_epoch=5, epochs=11, validation_data=test_generator, validation_steps=2)
# Retrieve training and validation accuracy
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
# Plot training and validation accuracy
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)