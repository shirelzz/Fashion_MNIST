from modelUtils import *
from fashionMNIST import *
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.keras import layers, models
layers, models = tf.keras.layers, tf.keras.models


def split_validation(X_train_, y_train_):
    X_train, X_validation, y_train, y_validation = train_test_split(X_train_, y_train_, test_size=0.1, random_state=42)
    return (X_train, X_validation, y_train, y_validation)


def neural_network(X_train, y_train, X_validation, y_validation, X_test, y_test):
    np.random.seed(42)  # random seed is used to replicate the same result every time
    tf.random.set_seed(42)
    model = models.Sequential()  # USING SEQUENTIAL API
    model.add(layers.Flatten(input_shape=[28 ,28]))  # input layer, converting 2D to 28*28 pixel using flatten

    # model.add(keras.layers.Dense(500, activation="relu")) # 1st hidden layer, 300=no. of neurons, relu=activation function

    # model.add(keras.layers.Dense(500, activation="relu")) # 1st hidden layer, 300=no. of neurons, relu=activation function
    model.add(layers.Dense(300, activation="relu")) # 1st hidden layer, 300=no. of neurons, relu=activation function
    model.add(layers.Dense(100,activation="relu")) # 2nd hidden layer
    model.add(layers.Dense(50,activation="relu")) # 3nd hidden layer
    model.add(layers.Dense(10,activation="sigmoid")) # output layer, categories=10

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    # keras.utils.plot_model(model, show_shapes=True)
    weights, biases = model.layers[1].get_weights()
    model.compile(loss="sparse_categorical_crossentropy" ,optimizer="sgd" ,metrics=["accuracy"])
    # scc = for categorical data
    # sgd = stochastic gradient descent
    # for binary labels = binary_cross_entropy
    X_train_shaped = X_train.values.reshape(-1, 28, 28)
    X_validation_shaped = X_validation.values.reshape(-1, 28, 28)
    X_test_shaped = X_test.values.reshape(-1, 28, 28)
    model_history = model.fit(X_train_shaped, y_train, epochs=50, batch_size=32, validation_data=(X_validation_shaped, y_validation))
    print("model_history.params", model_history.params)
    print("model_history.history", model_history.history)
    pd.DataFrame(model_history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
    print(model.evaluate(X_test_shaped, y_test))

    x_new = X_test_shaped[:]  # considering first three samples from test dataset
    y_proba = model.predict(x_new)  # probability to each class

    print("y_proba:  ", y_proba.round(2))
    y_pred = y_proba.argmax(axis=-1)
    print("y_pred:  ", y_pred.round(2))

    # as category starts from 0 to 9
    # for first record category = 9 which can be confirmed by above probabilities
    # category of second sample = 2
    # category of third sample = 1
