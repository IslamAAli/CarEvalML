from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix,classification_report
from sklearn import metrics

import Config
import plottingManagement

# ------------------------------------------------------------------------------------
def knn_train_cross_validate(X_train, y_train, X_valid, y_valid):
    knn_best_n_neighbors = 0
    knn_best_score = 0

    for i in range(Config.CFG_KNN_max_n):
        knn = KNeighborsClassifier(n_neighbors=i+1, weights='uniform', algorithm='auto')
        knn.fit(X_train, y_train)

        if Config.CFG_debug == 1:
            print('\nn= ', i+1, " - KNN Score = ", knn.score(X_train, y_train))

        if knn_best_score < knn.score(X_valid, y_valid):
            knn_best_score = knn.score(X_valid, y_valid)
            knn_best_n_neighbors = i+1

    return knn_best_score, knn_best_n_neighbors

# ------------------------------------------------------------------------------------
def knn_train_k_fold(X_labelled, y_labelled):
    knn_best_n_neighbors = 0
    knn_best_score = 0

    for i in range(Config.CFG_KNN_max_n):
        knn = KNeighborsClassifier(n_neighbors=i+1, weights='uniform', algorithm='auto')

        # begin k-fold loop
        k_fold_ranges = KFold(n_splits=Config.CFG_KNN_K_FOLD)
        k_fold_splits = k_fold_ranges.get_n_splits(X_labelled)
        itr_acc = 0
        for train_index, test_index in k_fold_ranges.split(X_labelled):
            X_train, X_valid = X_labelled[train_index], X_labelled[test_index]
            y_train, y_valid = y_labelled[train_index], y_labelled[test_index]

            # train the model with the k-fold training data
            knn.fit(X_train, y_train)

            if Config.CFG_debug == 1:
                print('\nn= ', i+1, " - KNN Score = ", knn.score(X_train, y_train))

            itr_acc += knn.score(X_valid, y_valid)

        itr_acc = itr_acc /k_fold_splits

        if knn_best_score < itr_acc:
            knn_best_score = itr_acc
            knn_best_n_neighbors = i+1

    return knn_best_score, knn_best_n_neighbors

# ------------------------------------------------------------------------------------
def knn_predict(n_best , X_labelled, y_labelled, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=n_best, weights='uniform', algorithm='auto')
    knn.fit(X_labelled, y_labelled)

    # compute the performance metrics
    y_predict       = knn.predict(X_test)
    confusion_mc    = confusion_matrix(y_test, y_predict)
    metrics_report  = classification_report(y_test, y_predict)

    # get the validation accuracy of the KNN algorithm
    return metrics.accuracy_score(y_test, y_predict), confusion_mc, metrics_report

# ------------------------------------------------------------------------------------
def knn_main(X_train_dec, y_train_dec, X_valid_dec, y_valid_dec,
            X_train_bin, y_train_bin, X_valid_bin, y_valid_bin,
            X_labelled_dec, y_labelled_dec, X_labelled_bin, y_labelled_bin,
            X_test_dec, y_test_dec, X_test_bin, y_test_bin,
            knn_validation_mode):

    if knn_validation_mode == Config.ValidationMethod.CROSS_VALIDATION:
        print('\n[1-1] K-Nearest Neighbor Classification - Cross Validation 3:1:1')
        print('====================================================================================')

        # get KNN algorithm training results
        knn_train_score_dec, knn_best_n_neighbors_dec = knn_train_cross_validate(X_train_dec, y_train_dec, X_valid_dec, y_valid_dec)
        knn_train_score_bin, knn_best_n_neighbors_bin = knn_train_cross_validate(X_train_bin, y_train_bin, X_valid_bin, y_valid_bin)

        # get KNN algorithm testing results
        knn_test_score_dec, knn_test_cm_dec, knn_test_mt_dec = knn_predict(knn_best_n_neighbors_dec, X_train_dec,y_train_dec, X_test_dec, y_test_dec)
        knn_test_score_bin, knn_test_cm_bin, knn_test_mt_bin = knn_predict(knn_best_n_neighbors_bin, X_train_bin, y_train_bin, X_test_bin, y_test_bin)
    else:
        print('\n[1-2] K-Nearest Neighbor Classification - k-fold Validation (k=', Config.CFG_KNN_K_FOLD, ')')
        print('====================================================================================')

        # get KNN algorithm training results
        knn_train_score_dec, knn_best_n_neighbors_dec = knn_train_k_fold(X_labelled_dec, y_labelled_dec)
        knn_train_score_bin, knn_best_n_neighbors_bin = knn_train_k_fold(X_labelled_bin, y_labelled_bin)

        knn_test_score_dec, knn_test_cm_dec, knn_test_mt_dec = knn_predict(knn_best_n_neighbors_dec, X_labelled_dec, y_labelled_dec, X_test_dec, y_test_dec)
        knn_test_score_bin, knn_test_cm_bin, knn_test_mt_bin = knn_predict(knn_best_n_neighbors_bin, X_labelled_bin, y_labelled_bin, X_test_bin, y_test_bin)

    # printing the validation score of the KNN training process
    print('=> Decimal Features: (KNN Validation Score: ', knn_train_score_dec, ' - n= ', knn_best_n_neighbors_dec, ')')
    print('=> Binary Features:  (KNN Validation Score: ', knn_train_score_bin, ' - n= ', knn_best_n_neighbors_bin, ')')

    print('=> Decimal Features: (KNN Testing Score: \t', knn_test_score_dec, ')')
    print('=> Binary Features:  (KNN Testing Score: \t', knn_test_score_bin, ')')

    Config.RES_VAL_KNN_DEC = knn_train_score_dec
    Config.RES_TEST_KNN_DEC = knn_test_score_dec
    Config.RES_VAL_KNN_BIN = knn_train_score_bin
    Config.RES_TEST_KNN_BIN = knn_test_score_bin
