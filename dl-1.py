import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
np.random.seed(42)
data = {'X': np.random.rand(100, 1).flatten()}
data['y']= 3 * data['X'] + 2 + 0.1 * np.random.randn(100)
df = pd.DataFrame(data)
print(df.info())
X_train, X_test, y_train, y_test = train_test_split(df['X'], df['y'], test_size=0.2, random_state=42)
#@title Build and Display Summary of Neural Network with Keras
model = Sequential()
# Add the output layer with 1 unit and linear activation
model.add(Dense(units=6, activation='relu'))
model.add(Dense(units=1, activation='linear'))
# Build the model
model.build(input_shape=(None, 1))
# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')
# Display the model summary
model.summary()
#@title Train, Predict, and Evaluate Neural Network
# Train the model
history = model.fit(X_train, y_train, epochs=100, verbose=0)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse:.4f}')
plt.scatter(X_test, y_test, color='black', label='Actual')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Scatter Plot of Actual vs Predicted Values (Simple Linear Regression)')
plt.legend()
plt.show()