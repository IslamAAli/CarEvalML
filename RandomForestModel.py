from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import metrics

import Config
import plottingManagement

# ------------------------------------------------------------------------------------
def rf_train(X_labelled, y_labelled):

    rf_parameters = {'n_estimators'     : Config.CFG_RF_N_ESTIMATORS,
                     'max_features'     : Config.CFG_RF_MAX_FEATURES,
                     'min_samples_leaf' : Config.CFG_RF_MIN_LEAF,
                     'criterion'        : Config.CFG_RF_CRITERION,
                     'max_depth'        : Config.CFG_RF_MAX_DEPTH,
                     'min_samples_split': Config.CFG_RF_MIN_SPLIT}

    rf = RandomForestClassifier()
    clf = RandomizedSearchCV(rf, rf_parameters, cv=None, n_jobs=3, refit=True)
    clf.fit(X_labelled, y_labelled)

    return clf.best_estimator_, clf.best_score_

# ------------------------------------------------------------------------------------
def rf_predict(rf, X_test, y_test):

    # compute the performance metrics
    y_predict       = rf.predict(X_test)
    confusion_mc    = confusion_matrix(y_test, y_predict)
    metrics_report  = classification_report(y_test, y_predict)

    # get the validation accuracy of the KNN algorithm
    return metrics.accuracy_score(y_test, y_predict), confusion_mc, metrics_report

# ------------------------------------------------------------------------------------
def rf_main(X_labelled_dec, y_labelled_dec, X_labelled_bin, y_labelled_bin,
            X_test_dec, y_test_dec, X_test_bin, y_test_bin):


    print('\n[4] Random Forest Classification - k-fold Validation (k=', Config.CFG_RF_K_FOLD, ')')
    print('====================================================================================')

    # get Random Forest algorithm training results
    rf_estimator_dec, rf_train_score_dec = rf_train(X_labelled_dec, y_labelled_dec)
    rf_estimator_bin, rf_train_score_bin = rf_train(X_labelled_bin, y_labelled_bin)

    # get Random Forest algorithm testing results
    rf_test_score_dec, rf_test_cm_dec, rf_test_mt_dec = rf_predict(rf_estimator_dec, X_test_dec, y_test_dec)
    rf_test_score_bin, rf_test_cm_bin, rf_test_mt_bin = rf_predict(rf_estimator_bin, X_test_bin, y_test_bin)

    # printing the validation score of the Random Forest training process
    print('=> Decimal Features: (Random Forest Validation Score: ', rf_train_score_dec, ')')
    print('=> Binary Features:  (Random Forest Validation Score: ', rf_train_score_bin, ')')

    print('=> Decimal Features: (Random Forest Testing Score: \t', rf_test_score_dec, ')')
    print('=> Binary Features:  (Random Forest Testing Score: \t', rf_test_score_bin, ')')

    # draw confusion matrix and print the metrics
    print('==> Decimal Features: Random Forest performance')
    print(rf_test_mt_dec)
    print('==> Binary Features: Random Forest performance')
    print(rf_test_mt_bin)
    plottingManagement.plot_confusion_matrix(rf_test_cm_dec, 'Random Forest Decimal Case')
    plottingManagement.plot_confusion_matrix(rf_test_cm_bin, 'Random Forest Binary Case')

    Config.RES_VAL_RAND_FOREST_DEC = rf_train_score_dec
    Config.RES_TEST_RAND_FOREST_DEC = rf_test_score_dec
    Config.RES_VAL_RAND_FOREST_BIN = rf_train_score_bin
    Config.RES_TEST_RAND_FOREST_BIN = rf_test_score_bin