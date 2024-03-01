from modelUtils import *
from fashionMNIST import *
from softMax import *
from RNN import *
from CNN import *


def main():
    df_shuffled = get_data()

    # For SoftMax, CNN
    X_train1, y_train1, X_test1, y_test1 = split_data1(df_shuffled)

    # For RNN
    X_train0, y_train0, X_test0, y_test0, X_validation0, y_validation0 = split_data0(df_shuffled)

    # Number of unique classes
    num_classes = len(np.unique(y_train1))

    # For showcasing the results of all models
    loss_array = []
    model_array = []
    accuracy_array = []

    # Perform training for softmax model
    print("SoftMax model:")
    softmax_model = train_softmax_model(X_train1, y_train1, X_test1, y_test1, num_classes)
    evaluate_model(softmax_model, X_test1, y_test1)

    # Perform training for CNN model
    print("CNN model:")
    cnn_model = train_cnn_model(X_train1, y_train1, X_test1, y_test1, num_classes)
    (loss, accuracy) = evaluate_model(cnn_model, X_test1.values.reshape(-1, 28, 28, 1), y_test1)
    loss_array.append(loss)
    model_array.append("CNN model")
    accuracy_array.append(accuracy)

    # Perform training for Neural network model
    print("Neural network model:")
    # (X_train, X_validation, y_train, y_validation) = split_validation(X_train_=X_train, y_train_=y_train)
    model = create_neural_network(num_classes)
    train_neural_network(model, X_train0, y_train0, X_test0, y_test0, X_validation0, y_validation0)
    (loss, accuracy) = evaluate_model(cnn_model, X_test0.values.reshape(-1, 28, 28, 1), y_test0)
    loss_array.append(loss)
    model_array.append("Neural network model")
    accuracy_array.append(accuracy)

    # Showcase results
    plt.figure(figsize=(20, 6))
    plt.bar(np.array(model_array), np.array(loss_array), color=['blue', 'orange', 'green'])
    plt.xlabel('Models')
    plt.ylabel('Loss function')
    plt.title('Comparison of loss function error among Models')
    plt.show()
    plt.savefig("imgFolder/lossFuncComparison")

    plt.figure(figsize=(20, 6))
    plt.bar(np.array(model_array), np.array(accuracy_array), color=['blue', 'orange', 'green'])
    plt.xlabel('Models')
    plt.ylabel('Accuracy score')
    plt.title('Comparison of accuracy score among Models')
    plt.show()
    plt.savefig("imgFolder/accurateFuncComparison")


main()
