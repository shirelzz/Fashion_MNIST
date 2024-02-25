from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
# import matplotlib.pyplot as plt
import importlib
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
import utils
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
import numpy as np
importlib.reload(utils)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
def get_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print(train_images)
    # Combine train and test images and labels
    images = np.concatenate((train_images, test_images))
    labels = np.concatenate((train_labels, test_labels))
    # Flatten images to 1D arrays
    images = images.reshape(-1, 28*28)
    # Convert images and labels to DataFrame
    data = {'image_' + str(i): images[:, i] for i in range(images.shape[1])}
    data['label'] = labels
    df = pd.DataFrame(data)
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    return df_shuffled

def split_data(df_shuffled):
    X = df_shuffled.drop('label', axis=1)  
    y = df_shuffled['label']
    X_train_, X_test, y_train_, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train_, y_train_, test_size=0.1, random_state=42)
    
    # To transform the images to be in scale from 0-1
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_validation = X_validation / 255.0
    return (X_train, y_train, X_test, y_test, X_validation, y_validation)

def find_best_k_for_KNN(X_train, y_train, X_test, y_test):
    k_range = range(1,50)
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
    print("The best accuracy of the knn model is when k =",best_k, ", and the score is:",max_score) 
    return best_k

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

def main():
    df_shuffled = get_data()
    (X_train, y_train, X_test, y_test, X_validation, y_validation) = split_data(df_shuffled)
    best_k = 5
    # best_k = find_best_k_for_KNN(X_train, y_train, X_test, y_test)
    
    knn = KNeighborsClassifier(n_neighbors=best_k)
    print("KNN model:")
    train_text_results_modules(knn, X_train, y_train, X_test, y_test)
    
    log_reg = LogisticRegression(max_iter=100000, random_state=42)
    print("LogisticRegression model:")
    train_text_results_modules(log_reg, X_train, y_train, X_test, y_test)
    
    randomForest = RandomForestClassifier()
    print("RandomForestClassifier model:")
    train_text_results_modules(randomForest, X_train, y_train, X_test, y_test)
    
    clf = SGDClassifier(random_state=42)
    print("SGDClassifier model:")
    train_text_results_modules(clf, X_train, y_train, X_test, y_test)
    
    sgd_clf = CalibratedClassifierCV(clf, cv=5, method='sigmoid')
    print("CalibratedClassifierCV model:")
    train_text_results_modules(sgd_clf, X_train, y_train, X_test, y_test)
    
    gnb = GaussianNB()
    print("GaussianNB model:")
    train_text_results_modules(gnb, X_train, y_train, X_test, y_test)
main()