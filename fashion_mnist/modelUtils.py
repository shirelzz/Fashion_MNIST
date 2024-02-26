import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def evaluate_model(model, X_test, y_test):
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    # Predict the classes for the test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))


def evaluate_model1(model, X_test, y_test):
    # Reshape the input data for CNN
    X_test = np.array(X_test).reshape(-1, 28, 28, 1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)


def train_text_results_modules(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    model.score(X_test, y_test)
    # MSE
    knn_mse = mean_squared_error(y_test, y_predict)
    print("MSE: ", knn_mse)

    # RMSE
    print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_predict)))

    # MAE
    print("MAE: ", np.sqrt(mean_absolute_error(y_test, y_predict)))

    # R2 Score
    print("R2 Score: ", r2_score(y_test, y_predict))


def find_best_k_for_KNN(X_train, y_train, X_test, y_test):
    k_range = range(1, 50)
    scores = []
    for k in k_range:
        print(k)
        knn_ = KNeighborsClassifier(n_neighbors=k)
        knn_.fit(X_train, y_train)
        y_pred = knn_.predict(X_test)
        scores.append(metrics.accuracy_score(y_test, y_pred))

    # plt.figure(figsize=(12, 6))
    # plt.plot(k_range, scores,color='red', linestyle='dashed', marker='o',
    #         markerfacecolor='blue', markersize=10)
    # plt.xlabel('Value of K for KNN')
    # plt.ylabel('Testing Accuracy')

    # Finding the maximum k - the number of nearest neighbors:
    max_score = max(scores)
    best_k = scores.index(max_score)
    print("The best accuracy of the knn model is when k =", best_k, ", and the score is:", max_score)

    return best_k
