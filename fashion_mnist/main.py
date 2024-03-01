from modelUtils import *
from fashionMNIST import *
from softMax import *
from RNN import *
from CNN import *
from sklearn.neighbors import KNeighborsClassifier


def main():
    df_shuffled = get_data()

    # For RNN
    X_train0, y_train0, X_test0, y_test0, X_validation0, y_validation0 = split_data0(df_shuffled)

    # For SoftMax, CNN
    X_train1, y_train1, X_test1, y_test1 = split_data1(df_shuffled)

    num_classes = len(np.unique(y_train1))  # Number of unique classes

    best_k = 5
    loss_array = []
    model_array = []
    accuracy_array = []
    # best_k = find_best_k_for_KNN(X_train1, y_train1, X_test1, y_test1)

    # knn = KNeighborsClassifier(n_neighbors=best_k)
    # print("KNN model:")
    # (loss, accuracy) = train_test_results_modules(knn, X_train1, y_train1, X_test1, y_test1)
    # loss_array.append(loss)
    # model_array.append("KNN")
    # accuracy_array.append(accuracy)
    
    # log_reg = LogisticRegression(max_iter=100000, random_state=42)
    # print("LogisticRegression model:")
    # (loss, accuracy) = train_test_results_modules(log_reg, X_train1, y_train1, X_test1, y_test1)
    # loss_array.append(loss)
    # model_array.append("LogisticRegression")
    # accuracy_array.append(accuracy)
    
    # randomForest = RandomForestClassifier()
    # print("RandomForestClassifier model:")
    # (loss, accuracy) = train_test_results_modules(randomForest, X_train1, y_train1, X_test1, y_test1)
    # loss_array.append(loss)
    # model_array.append("RandomForestClassifier")
    # accuracy_array.append(accuracy)
    
    # clf = SGDClassifier(random_state=42)
    # print("SGDClassifier model:")
    # (loss, accuracy) = train_test_results_modules(clf, X_train1, y_train1, X_test1, y_test1)
    # loss_array.append(loss)
    # model_array.append("SGDClassifier")
    # accuracy_array.append(accuracy)
    
    # sgd_clf = CalibratedClassifierCV(clf, cv=5, method='sigmoid')
    # print("CalibratedClassifierCV model:")
    # (loss, accuracy) = train_test_results_modules(sgd_clf, X_train1, y_train1, X_test1, y_test1)
    # loss_array.append(loss)
    # model_array.append("CalibratedClassifierCV")
    # accuracy_array.append(accuracy)
    
    # gnb = GaussianNB()
    # print("GaussianNB model:")
    # (loss, accuracy) = train_test_results_modules(gnb, X_train1, y_train1, X_test1, y_test1)
    # loss_array.append(loss)
    # model_array.append("GaussianNB")
    # accuracy_array.append(accuracy)
    
    # Perform training for softmax model

    print("SoftMax model:")
    softmax_model = train_softmax_model(X_train1, y_train1, X_test1, y_test1, num_classes)
    (loss, accuracy) = evaluate_model(softmax_model, X_test1, y_test1)
    loss_array.append(loss)
    model_array.append("SoftMax model")
    accuracy_array.append(accuracy)
    
    # Perform training for CNN model
    print("CNN model:")
    cnn_model = train_cnn_model(X_train1, y_train1, X_test1, y_test1, num_classes)
    (loss, accuracy) = evaluate_model(cnn_model, X_test1.values.reshape(-1, 28, 28, 1), y_test1)
    loss_array.append(loss)
    model_array.append("CNN model")
    accuracy_array.append(accuracy)
    
    # Perform training for RNN model
    print("Neural network model:")
    # (X_train, X_validation, y_train, y_validation) = split_validation(X_train_=X_train, y_train_=y_train)
    model = create_neural_network(num_classes)
    train_neural_network(model, X_train0, y_train0, X_test0, y_test0, X_validation0, y_validation0)
    (loss, accuracy) = evaluate_model(model, X_test0.values.reshape(-1, 28, 28, 1), y_test0)
    loss_array.append(loss)
    model_array.append("Neural network model")
    accuracy_array.append(accuracy)
    
    
    plt.figure(figsize=(8, 5))
    plt.bar(np.array(model_array), np.array(loss_array), color=['blue', 'orange', 'green'])
    plt.xlabel('Models')
    plt.ylabel('Loss function')
    plt.title('Comparison of loss function error among Models')
    plt.savefig("imgFolder/lossFuncComparison")
    plt.show()
    
    plt.figure(figsize=(8, 5))
    plt.bar(np.array(model_array), np.array(accuracy_array), color=['blue', 'orange', 'green'])
    plt.xlabel('Models')
    plt.ylabel('Accuracy score')
    plt.title('Comparison of accuracy score among Models')
    plt.savefig("imgFolder/accurateFuncComparison")
    plt.show()


main()

