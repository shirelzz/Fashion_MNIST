import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

layers, models = tf.keras.layers, tf.keras.models


def create_softmax_model(input_shape, num_classes):
    """
    Creates a softmax regression model.

    Parameters:
    - input_shape: Tuple specifying the shape of the input images (height, width, channels).
    - num_classes: Number of classes for classification.

    Returns:
    - model: A model object.
    """

    model = models.Sequential([
        layers.Dense(num_classes, activation='softmax', input_shape=input_shape)
    ])

    return model


def train_softmax_model(X_train, y_train, X_test, y_test, num_classes):
    """
    Trains a softmax regression model.

    Parameters:
    - X_train: Training input data.
    - y_train: Training labels.
    - X_test: Test input data.
    - y_test: Test labels.
    - num_classes: Number of classes for classification.

    Returns:
    - model: Trained softmax regression model.

    """

    input_shape = X_train.shape[1:]

    # Create the model
    model = create_softmax_model(input_shape, num_classes)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    epochs = 20
    model_history = model.fit(X_train, y_train,
                              epochs=epochs,
                              batch_size=32,
                              validation_data=(X_test, y_test))

    # Display the training history
    pd.DataFrame(model_history.history).plot(figsize=(8, 5))
    plt.title('Softmax Training History \n epoch: ' + str(epochs))
    plt.show()

    return model
