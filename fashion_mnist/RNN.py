from modelUtils import *
from fashionMNIST import *
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.keras import layers, models
layers, models, callbacks = tf.keras.layers, tf.keras.models, tf.keras.callbacks


def split_validation(X_train_, y_train_):
    X_train, X_validation, y_train, y_validation = train_test_split(X_train_, y_train_, test_size=0.1, random_state=42)
    return (X_train, X_validation, y_train, y_validation)


def create_neural_network(num_classes): # create_neural_network = 10
    model_input = layers.Input(shape=[28 ,28])
    x = layers.Flatten()(model_input)  # Flatten input
    x = layers.Dense(300, activation="relu")(x)  # 1st hidden layer
    # x = layers.Dense(100, activation="relu")(x)  # 2nd hidden layer
    x = layers.Dense(128, activation="relu")(x)   # 3rd hidden layer
    model_output = layers.Dense(num_classes, activation="softmax")(x)  # Output layer

    model = models.Model(inputs=model_input, outputs=model_output)
    return model

def train_neural_network(model, X_train, y_train, X_test, y_test, x_validation, y_validation):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    X_train_shaped = X_train.values.reshape(-1, 28, 28)
    X_validation_shaped = x_validation.values.reshape(-1, 28, 28)
    X_test_shaped = X_test.values.reshape(-1, 28, 28)
    
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    epochs=30
    model_history = model.fit(X_train_shaped, 
                              y_train,
                              epochs=epochs,
                              validation_data=(X_validation_shaped, y_validation),
                              callbacks=[early_stopping])

    stop = early_stopping.stopped_epoch
    if stop == 0:
        stop = epochs
    pd.DataFrame(model_history.history).plot(figsize=(8, 5))
    plt.title('Simple Neural Network Training History \n epoch: ' + str(stop))
    plt.gca().set_ylim(0, 1)
    plt.savefig("imgFolder/simpleNeuralNetwork_fig")
    plt.show()

    model.evaluate(X_test_shaped, y_test)
    