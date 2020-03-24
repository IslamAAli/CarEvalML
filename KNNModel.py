from sklearn.neighbors import KNeighborsClassifier
import Config


def knn_train(X_train, y_train, X_valid, y_valid):
    knn_best_n_neighbors = 0
    knn_best_score = 0

    for i in range(Config.CFG_KNN_max_n):
        knn = KNeighborsClassifier(n_neighbors=i+1)
        knn.fit(X_train, y_train)

        if Config.CFG_debug == 1:
            print('\nn= ', i+1, " - KNN Score = ", knn.score(X_train, y_train))

        if knn_best_score < knn.score(X_valid, y_valid):
            knn_best_score = knn.score(X_valid, y_valid)
            knn_best_n_neighbors = i+1

    return knn_best_score, knn_best_n_neighbors


def knn_test(n_best , X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=n_best)
    knn.fit(X_train, y_train)

    # get the validation accuracy of the KNN algorithm
    return knn.score(X_test, y_test)
