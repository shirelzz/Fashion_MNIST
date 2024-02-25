from modelUtils import *
from fashionMNIST import *
from softMax import *
from RNN import *
from sklearn.neighbors import KNeighborsClassifier


def main():
    df_shuffled = get_data()
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(df_shuffled)
    best_k = 5
    # best_k = find_best_k_for_KNN(X_train, y_train, X_test, y_test)

    # knn = KNeighborsClassifier(n_neighbors=best_k)
    # print("KNN model:")
    # train_text_results_modules(knn, X_train, y_train, X_test, y_test)
    #
    # log_reg = LogisticRegression(max_iter=100000, random_state=42)
    # print("LogisticRegression model:")
    # train_text_results_modules(log_reg, X_train, y_train, X_test, y_test)
    #
    # randomForest = RandomForestClassifier()
    # print("RandomForestClassifier model:")
    # train_text_results_modules(randomForest, X_train, y_train, X_test, y_test)
    #
    # clf = SGDClassifier(random_state=42)
    # print("SGDClassifier model:")
    # train_text_results_modules(clf, X_train, y_train, X_test, y_test)
    #
    # sgd_clf = CalibratedClassifierCV(clf, cv=5, method='sigmoid')
    # print("CalibratedClassifierCV model:")
    # train_text_results_modules(sgd_clf, X_train, y_train, X_test, y_test)
    #
    # gnb = GaussianNB()
    # print("GaussianNB model:")
    # train_text_results_modules(gnb, X_train, y_train, X_test, y_test)

    print("SoftMax model:")
    # train_softmax_regression(X_train, y_train, X_test, y_test)
    num_classes = len(np.unique(y_train))  # Number of unique classes

    # Perform training for softmax model with hidden layers
    softmax_model = train_softmax_model(X_train, y_train, X_test, y_test, num_classes)
    evaluate_model(softmax_model, X_test, y_test)

    # print("RNN model:")
    # (X_train, X_validation, y_train, y_validation) = split_validation(X_train_=X_train, y_train_=y_train)
    # neural_network(X_train, y_train, X_validation, y_validation, X_test, y_test)


main()

