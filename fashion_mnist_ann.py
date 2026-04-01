# Fashion MNIST Classification using ANN

# 1. Import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 2. Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# 3. Preprocessing
# Normalize (0–255 → 0–1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten images (28x28 → 784)
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# 4. Build model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

# 5. Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# 7. Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)

print(f"\nTest Accuracy: {accuracy:.4f}")