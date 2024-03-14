import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
import matplotlib.pyplot as plt
# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
# Define LeNet-5 architecture
model = Sequential([
Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=(28, 28, 1)),
AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(16, kernel_size=(5, 5), activation='tanh'),
AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
Flatten(),
Dense(120, activation='tanh'),
Dense(84, activation='tanh'),
Dense(10, activation='softmax')
])
# Compile model
model.compile(loss='categorical_crossentropy',
optimizer='adam',
metrics=['accuracy'])
# Train model
history = model.fit(x_train, y_train,
batch_size=128, epochs=12,
validation_data=(x_test, y_test))
# Evaluate model
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss score: {score[0]}')
print(f'Test accuracy score:{ score[1]}')
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()