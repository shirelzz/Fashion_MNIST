import matplotlib.pyplot as plt
from modelUtils import *
import pandas as pd
import tensorflow as tf
# from tensorflow.keras import layers, models
layers, models = tf.keras.layers, tf.keras.models


def create_softmax_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_softmax_model(X_train, y_train, X_test, y_test, num_classes):
    input_shape = X_train.shape[1:]

    # Create the model
    model = create_softmax_model(input_shape, num_classes)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    epochs = 10
    # Train the model
    model_history = model.fit(X_train, y_train, epochs= epochs, batch_size=32, validation_data=(X_test, y_test))
    # model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))

    # Display the training history
    pd.DataFrame(model_history.history).plot(figsize=(8, 5))
    plt.title('Soft max Training History, epochs: ' + str(epochs))
    plt.show()
    plt.savefig("imgFolder/softMax_fig")
    return model
