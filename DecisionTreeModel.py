from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix,classification_report
from sklearn import metrics

import Config
import plottingManagement

# ------------------------------------------------------------------------------------
def dt_train_cross_validate(X_train, y_train, X_valid, y_valid):

    dt_best_criterion           = 'entropy'
    dt_best_max_depth           = None
    dt_best_min_samples_split   = 2
    dt_best_min_samples_leaf    = 1
    dt_best_score               = 0

    for dt_c in Config.CFG_DT_CRITERION:
        for dt_max_d in Config.CFG_DT_MAX_DEPTH:
            for dt_min_s in Config.CFG_DT_MIN_SPLIT:
                for dt_min_l in Config.CFG_DT_MIN_LEAF:
                    dt = tree.DecisionTreeClassifier(criterion=dt_c,
                                                     max_depth=dt_max_d,
                                                     min_samples_split=dt_min_s,
                                                     min_samples_leaf=dt_min_l)

                    dt.fit(X_train, y_train)
                    valid_score = dt.score(X_valid, y_valid)

                    if dt_best_score < valid_score:
                        dt_best_score               = valid_score
                        dt_best_criterion           = dt_c
                        dt_best_max_depth           = dt_max_d
                        dt_best_min_samples_split   = dt_min_s
                        dt_best_min_samples_leaf    = dt_min_l


    return dt_best_score, dt_best_criterion, dt_best_max_depth, dt_best_min_samples_split, dt_best_min_samples_leaf

# ------------------------------------------------------------------------------------
def dt_train_k_fold(X_labelled, y_labelled):

    dt_best_criterion           = 'entropy'
    dt_best_max_depth           = None
    dt_best_min_samples_split   = 2
    dt_best_min_samples_leaf    = 1
    dt_best_score               = 0

    for dt_c in Config.CFG_DT_CRITERION:
        for dt_max_d in Config.CFG_DT_MAX_DEPTH:
            for dt_min_s in Config.CFG_DT_MIN_SPLIT:
                for dt_min_l in Config.CFG_DT_MIN_LEAF:
                    dt = tree.DecisionTreeClassifier(criterion=dt_c,
                                                     max_depth=dt_max_d,
                                                     min_samples_split=dt_min_s,
                                                     min_samples_leaf=dt_min_l)

                    # begin k-fold loop
                    k_fold_ranges = KFold(n_splits=Config.CFG_KNN_K_FOLD)
                    k_fold_splits = k_fold_ranges.get_n_splits(X_labelled)
                    itr_acc = 0
                    for train_index, test_index in k_fold_ranges.split(X_labelled):
                        X_train, X_valid = X_labelled[train_index], X_labelled[test_index]
                        y_train, y_valid = y_labelled[train_index], y_labelled[test_index]

                        # train the model with the k-fold training data
                        dt.fit(X_train, y_train)

                        itr_acc += dt.score(X_valid, y_valid)

                    itr_acc = itr_acc / k_fold_splits

                    if dt_best_score < itr_acc:
                        dt_best_score               = itr_acc
                        dt_best_criterion           = dt_c
                        dt_best_max_depth           = dt_max_d
                        dt_best_min_samples_split   = dt_min_s
                        dt_best_min_samples_leaf    = dt_min_l


    return dt_best_score, dt_best_criterion, dt_best_max_depth, dt_best_min_samples_split, dt_best_min_samples_leaf

# ------------------------------------------------------------------------------------
def dt_predict(best_criterion, best_max_depth, best_min_samples_split, best_min_samples_leaf , X_labelled, y_labelled, X_test, y_test):
    dt = tree.DecisionTreeClassifier(criterion=best_criterion,
                                     max_depth=best_max_depth,
                                     min_samples_split=best_min_samples_split,
                                     min_samples_leaf=best_min_samples_leaf)
    dt.fit(X_labelled, y_labelled)

    # compute the performance metrics
    y_predict       = dt.predict(X_test)
    confusion_mc    = confusion_matrix(y_test, y_predict)
    metrics_report  = classification_report(y_test, y_predict)

    # get the validation accuracy of the KNN algorithm
    return metrics.accuracy_score(y_test, y_predict), confusion_mc, metrics_report

# ------------------------------------------------------------------------------------
def dt_main(X_train_dec, y_train_dec, X_valid_dec, y_valid_dec,
            X_train_bin, y_train_bin, X_valid_bin, y_valid_bin,
            X_labelled_dec, y_labelled_dec, X_labelled_bin, y_labelled_bin,
            X_test_dec, y_test_dec, X_test_bin, y_test_bin,
            dt_validation_mode):

    if dt_validation_mode == Config.ValidationMethod.CROSS_VALIDATION:
        print('\n[3-1] Decision Tree Classification - Cross Validation 3:1:1')
        print('====================================================================================')

        # get KNN algorithm training results
        dt_train_score_dec, \
        dt_best_criterion_dec, \
        dt_best_max_depth_dec, \
        dt_best_min_samples_split_dec, \
        dt_best_min_samples_leaf_dec = dt_train_cross_validate(X_train_dec, y_train_dec, X_valid_dec, y_valid_dec)

        dt_train_score_bin, \
        dt_best_criterion_bin, \
        dt_best_max_depth_bin, \
        dt_best_min_samples_split_bin, \
        dt_best_min_samples_leaf_bin = dt_train_cross_validate(X_train_bin, y_train_bin, X_valid_bin, y_valid_bin)

        # get KNN algorithm testing results
        dt_test_score_dec, dt_test_cm_dec, dt_test_mt_dec = dt_predict(dt_best_criterion_dec,
                                                                       dt_best_max_depth_dec,
                                                                       dt_best_min_samples_split_dec,
                                                                       dt_best_min_samples_leaf_dec,
                                                                       X_train_dec,y_train_dec, X_test_dec, y_test_dec)

        dt_test_score_bin, dt_test_cm_bin, dt_test_mt_bin = dt_predict(dt_best_criterion_bin,
                                                                       dt_best_max_depth_bin,
                                                                       dt_best_min_samples_split_bin,
                                                                       dt_best_min_samples_leaf_bin,
                                                                       X_train_bin, y_train_bin, X_test_bin, y_test_bin)
    else:
        print('\n[3-2] Decision Tree Classification - k-fold Validation (k=', Config.CFG_DT_K_FOLD, ')')
        print('====================================================================================')

        # get KNN algorithm training results
        dt_train_score_dec, \
        dt_best_criterion_dec, \
        dt_best_max_depth_dec, \
        dt_best_min_samples_split_dec, \
        dt_best_min_samples_leaf_dec = dt_train_k_fold(X_labelled_dec, y_labelled_dec)

        dt_train_score_bin, \
        dt_best_criterion_bin, \
        dt_best_max_depth_bin, \
        dt_best_min_samples_split_bin, \
        dt_best_min_samples_leaf_bin = dt_train_k_fold(X_labelled_bin, y_labelled_bin)

        # get KNN algorithm testing results
        dt_test_score_dec, dt_test_cm_dec, dt_test_mt_dec = dt_predict(dt_best_criterion_dec,
                                                                       dt_best_max_depth_dec,
                                                                       dt_best_min_samples_split_dec,
                                                                       dt_best_min_samples_leaf_dec,
                                                                       X_labelled_dec, y_labelled_dec, X_test_dec, y_test_dec)

        dt_test_score_bin, dt_test_cm_bin, dt_test_mt_bin = dt_predict(dt_best_criterion_bin,
                                                                       dt_best_max_depth_bin,
                                                                       dt_best_min_samples_split_bin,
                                                                       dt_best_min_samples_leaf_bin,
                                                                       X_labelled_bin, y_labelled_bin, X_test_bin, y_test_bin)

    # printing the validation score of the KNN training process
    print('=> Decimal Features: (Decision Tree Validation Score: ', dt_train_score_dec, ')')
    print('=> Binary Features:  (Decision Tree Validation Score: ', dt_train_score_bin, ')')

    print('=> Decimal Features: (Decision Tree Testing Score: \t', dt_test_score_dec, ')')
    print('=> Binary Features:  (Decision Tree Testing Score: \t', dt_test_score_bin, ')')