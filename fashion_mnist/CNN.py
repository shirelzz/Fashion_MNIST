import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

layers, models = tf.keras.layers, tf.keras.models
BatchNormalization, Dropout, LeakyReLU, EarlyStopping = tf.keras.layers.BatchNormalization,\
                                                        tf.keras.layers.Dropout,\
                                                        tf.keras.layers.LeakyReLU,\
                                                        tf.keras.callbacks.EarlyStopping


def create_cnn_model(input_shape, num_classes):

    """

    Description:
    This function constructs a CNN model using the Keras Sequential API.
    The model architecture consists of several layers.
    A more detailed explanation for each layer or step
    in this function can be found in the article.

    Parameters:
    - input_shape: Tuple specifying the shape of the input images (height, width, channels).
    - num_classes: Integer indicating the number of classes in the classification task.

    Returns:
    - model: A model object.

    """
    activation = 'relu'
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
    This function prepares the training and test datasets by reshaping them into the format
    required for CNN input, constructs the CNN model by calling the `create_cnn_model` function,
    compiles the model and trains the model using the provided training data.
    I used early stopping to halt training if the validation loss does not improve for three consecutive epochs,
    trying to avoid overfitting.
    The training process's history is plotted, showing metrics evolution over epochs.

    Parameters:
    - X_train: training data features.
    - y_train: training data labels.
    - X_test: test data features.
    - y_test: test data labels.
    - num_classes: number of classes in the dataset.

    Returns:
    - model: The trained model object.

    The model expects input data in the shape of 28x28 pixel grayscale images and is tailored
    for a classification task with a specified number of classes. After training, the function
    plots the training and validation accuracy and loss metrics over epochs and returns the
    trained model for further use or evaluation.
    """
    # Reshape the input data for CNN
    X_train = np.array(X_train).reshape(-1, 28, 28, 1)
    X_test = np.array(X_test).reshape(-1, 28, 28, 1)

    # Create the model
    model = create_cnn_model((28, 28, 1), num_classes)  # grayscale image of size 28x28 pixels with one channel.

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    epochs = 20
    model_history = model.fit(X_train, y_train,
                              epochs=epochs,
                              batch_size=32,
                              validation_data=(X_test, y_test),
                              callbacks=[early_stopping])

    # Get the epoch where the fit stopped in
    stop = early_stopping.stopped_epoch

    # Display the training history
    pd.DataFrame(model_history.history).plot(figsize=(8, 5))
    plt.title('CNN Training History \n epoch: ' + str(stop))
    plt.show()

    return model




