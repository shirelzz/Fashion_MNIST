import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
# from keras.src.layers import LeakyReLU

# from tensorflow.keras import layers, models
layers, models = tf.keras.layers, tf.keras.models
BatchNormalization, Dropout, LeakyReLU = tf.keras.layers.BatchNormalization, tf.keras.layers.Dropout, tf.keras.layers.LeakyReLU

def create_cnn_model(input_shape, num_classes):

    """

    Description:
    This function constructs a CNN model using the Keras Sequential API.
    The model architecture consists of several layers:

    Convolutional Layers: These layers apply a convolution operation to the input image,
    extracting various features through filters.
    The activation function used is ReLU,
    which introduces non-linearity into the model, enabling it to learn complex patterns.

    Batch Normalization: This technique is used to improve the training stability and speed
    by normalizing the inputs of each layer.

    MaxPooling Layers: Max pooling reduces the spatial dimensions of the feature maps,
    effectively downsampling the input.
    It retains the most important features while reducing computational
    complexity and preventing overfitting.

    Flatten Layer: This layer flattens the 2D feature maps into a 1D vector,
    preparing the data for input into a fully connected neural network.

    Dense Layers: Fully connected layers that perform classification based on the learned features.
    The first dense layer has 64 units with ReLU activation, enabling the network to learn complex
    patterns in the flattened feature vectors.
    The final dense layer has units equal to the number of classes
    in the dataset, with softmax activation, which outputs probabilities for each class,
    indicating the likelihood of the input image belonging to each class.

    Dropout: This layer was added to prevent overfitting.
    It works by randomly setting a fraction of input units to zero during training,
    which helps to prevent the network from relying too much on specific neurons
    and encourages it to learn more robust features.
    We've defined a Dropout layer with a dropout rate of 0.2.
    This means during training, 20% of the input units to the Dropout layer will be randomly set to zero.

    Parameters:
    - input_shape: Tuple specifying the shape of the input images (height, width, channels).
    - num_classes: Integer indicating the number of classes in the classification task.

    notes:
        - I was experiencing with ReLU and Leaky ReLU and the results were slightly better with ReLU.
        - Deepening the layers more with MaxPooling2D is impossible because the input images are relatively small (28*28)

    """
    # activation = 'relu'
    activation = LeakyReLU(alpha=0.1)
    dropout_rate = 0.3

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation=activation, input_shape=input_shape),  # 32 filters of size 3x3
        BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation=activation),
        BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation=activation),
        BatchNormalization(),
        layers.Flatten(),
        layers.Dense(64, activation=activation),
        Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def train_cnn_model(X_train, y_train, X_test, y_test, num_classes):
    """
    notes: I was experiencing with different parameters:
    - number of epochs: 8, 10, 12, 16
      results were best with 8
    - optimizers: Adam, SGD, RMSprop
      results were best with Adam
    - loss functions:

    """
    # Reshape the input data for CNN
    X_train = np.array(X_train).reshape(-1, 28, 28, 1)
    X_test = np.array(X_test).reshape(-1, 28, 28, 1)

    # Create the model
    model = create_cnn_model((28, 28, 1), num_classes)  # grayscale image of size 28x28 pixels with one channel.

    # Compile the model
    # optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.001, momentum=0.9)
    optimizer = 'adam'
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    epochs = 8
    model_history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))

    # Display the training history
    pd.DataFrame(model_history.history).plot(figsize=(8, 5))
    plt.title('CNN Training History, epochs: ' + str(epochs))
    plt.show()

    return model




