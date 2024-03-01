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

    num_classes = len(np.unique(y_train1))  # Number of unique classes

    # Perform training for softmax model
    print("SoftMax model:")
    softmax_model = train_softmax_model(X_train1, y_train1, X_test1, y_test1, num_classes)
    evaluate_model(softmax_model, X_test1, y_test1)

    # Perform training for CNN model
    print("CNN model:")
    cnn_model = train_cnn_model(X_train1, y_train1, X_test1, y_test1, num_classes)
    evaluate_model1(cnn_model, X_test1, y_test1)

    # Perform training for RNN model
    print("RNN model:")
    neural_network(X_train0, y_train0, X_validation0, y_validation0, X_test0, y_test0)


main()
