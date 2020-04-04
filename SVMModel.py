from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import metrics

import Config

# ------------------------------------------------------------------------------------
def svm_train(X_labelled, y_labelled):

    svm_parameters = {'C'     : Config.CFG_SVM_C,
                      'kernel': Config.CFG_SVM_KERNEL,
                      'degree': Config.CFG_SVM_DEGREE,
                      'gamma' : Config.CFG_SVM_GAMMA}

    svm = SVC()
    clf = RandomizedSearchCV(svm, svm_parameters, cv=None, n_jobs=-1, refit=True)
    clf.fit(X_labelled, y_labelled)

    return clf.best_estimator_, clf.best_score_

# ------------------------------------------------------------------------------------
def svm_predict(svm, X_test, y_test):

    # compute the performance metrics
    y_predict       = svm.predict(X_test)
    confusion_mc    = confusion_matrix(y_test, y_predict)
    metrics_report  = classification_report(y_test, y_predict)

    # get the validation accuracy of the KNN algorithm
    return metrics.accuracy_score(y_test, y_predict), confusion_mc, metrics_report

# ------------------------------------------------------------------------------------
def svm_main(X_labelled_dec, y_labelled_dec, X_labelled_bin, y_labelled_bin,
                X_test_dec, y_test_dec, X_test_bin, y_test_bin):


    print('\n[5] SVM Classification - k-fold Validation (k=', Config.CFG_SVM_K_FOLD, ')')
    print('====================================================================================')

    # get Random Forest algorithm training results
    svm_estimator_dec, svm_train_score_dec = svm_train(X_labelled_dec, y_labelled_dec)
    svm_estimator_bin, svm_train_score_bin = svm_train(X_labelled_bin, y_labelled_bin)

    # get Random Forest algorithm testing results
    svm_test_score_dec, svm_test_cm_dec, svm_test_mt_dec = svm_predict(svm_estimator_dec, X_test_dec, y_test_dec)
    svm_test_score_bin, svm_test_cm_bin, svm_test_mt_bin = svm_predict(svm_estimator_bin, X_test_bin, y_test_bin)

    # printing the validation score of the Random Forest training process
    print('=> Decimal Features: (SVM Validation Score: ', svm_train_score_dec, ')')
    print('=> Binary Features:  (SVM Validation Score: ', svm_train_score_bin, ')')

    print('=> Decimal Features: (SVM Testing Score: \t', svm_test_score_dec, ')')
    print('=> Binary Features:  (SVM Testing Score: \t', svm_test_score_bin, ')')

    Config.RES_VAL_SVM_DEC = svm_train_score_dec
    Config.RES_TEST_SVM_DEC = svm_test_score_dec
    Config.RES_VAL_SVM_BIN = svm_train_score_bin
    Config.RES_TEST_SVM_BIN = svm_test_score_bin