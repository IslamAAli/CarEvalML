from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import metrics

import Config
import plottingManagement

# ------------------------------------------------------------------------------------
def dt_train(X_labelled, y_labelled):

    dt_parameters = {'min_samples_leaf' : Config.CFG_DT_MIN_LEAF,
                     'criterion'        : Config.CFG_DT_CRITERION,
                     'max_depth'        : Config.CFG_DT_MAX_DEPTH,
                     'min_samples_split': Config.CFG_DT_MIN_SPLIT}

    dt = tree.DecisionTreeClassifier()
    clf = RandomizedSearchCV(dt, dt_parameters, cv=None, n_jobs=3, refit=True)
    clf.fit(X_labelled, y_labelled)

    print('**==> Decision Tree Hyperparameters')
    print(clf.best_params_)

    return clf.best_estimator_, clf.best_score_

# ------------------------------------------------------------------------------------
def dt_predict(dt, X_test, y_test):

    # compute the performance metrics
    y_predict       = dt.predict(X_test)
    confusion_mc    = confusion_matrix(y_test, y_predict)
    metrics_report  = classification_report(y_test, y_predict)

    # get the validation accuracy of the KNN algorithm
    return metrics.accuracy_score(y_test, y_predict), confusion_mc, metrics_report

# ------------------------------------------------------------------------------------
def dt_main(X_labelled_dec, y_labelled_dec, X_labelled_bin, y_labelled_bin,
            X_test_dec, y_test_dec, X_test_bin, y_test_bin):


    print('\n[3] Decision Tree Classification - k-fold Validation (k=', Config.CFG_DT_K_FOLD, ')')
    print('====================================================================================')

    # get Random Forest algorithm training results
    dt_estimator_dec, dt_train_score_dec = dt_train(X_labelled_dec, y_labelled_dec)
    dt_estimator_bin, dt_train_score_bin = dt_train(X_labelled_bin, y_labelled_bin)

    # get Random Forest algorithm testing results
    dt_test_score_dec, dt_test_cm_dec, dt_test_mt_dec = dt_predict(dt_estimator_dec, X_test_dec, y_test_dec)
    dt_test_score_bin, dt_test_cm_bin, dt_test_mt_bin = dt_predict(dt_estimator_bin, X_test_bin, y_test_bin)

    # printing the validation score of the Random Forest training process
    print('=> Decimal Features: (Decision Tree Validation Score: ', dt_train_score_dec, ')')
    print('=> Binary Features:  (Decision Tree Validation Score: ', dt_train_score_bin, ')')

    print('=> Decimal Features: (Decision Tree Testing Score: \t', dt_test_score_dec, ')')
    print('=> Binary Features:  (Decision Tree Testing Score: \t', dt_test_score_bin, ')')

    # draw confusion matrix and print the metrics
    print('==> Decimal Features: Decision Tree performance')
    print(dt_test_mt_dec)
    print('==> Binary Features: Decision Tree performance')
    print(dt_test_mt_bin)
    plottingManagement.plot_confusion_matrix(dt_test_cm_dec, 'Decision Tree Decimal Case')
    plottingManagement.plot_confusion_matrix(dt_test_cm_bin, 'Decision Tree Binary Case')

    Config.RES_VAL_DEC_TREE_DEC = dt_train_score_dec
    Config.RES_TEST_DEC_TREE_DEC = dt_test_score_dec
    Config.RES_VAL_DEC_TREE_BIN = dt_train_score_bin
    Config.RES_TEST_DEC_TREE_BIN = dt_test_score_bin