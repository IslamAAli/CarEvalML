from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix,classification_report
from sklearn import metrics

import Config
import plottingManagement

# ------------------------------------------------------------------------------------
def rf_train_cross_validate(X_train, y_train, X_valid, y_valid):

    rf_best_criterion           = 'gini'
    rf_best_max_depth           = None
    rf_best_min_sample_split    = 2
    rf_best_n_estimators        = 'entropy'
    rf_best_max_features        = None
    rf_best_min_samples_leaf    = 1
    rf_best_score               = 0

    for rf_n in Config.CFG_RF_N_ESTIMATORS:
        for rf_max_f in Config.CFG_RF_MAX_FEATURES:
            for rf_min_l in Config.CFG_RF_MIN_LEAF:
                for rf_c in Config.CFG_RF_CRITERION:
                    for rf_max_d in Config.CFG_RF_MAX_DEPTH:
                        for rf_min_ss in Config.CFG_RF_MIN_SPLIT:
                            rf = RandomForestClassifier(n_estimators=rf_n,
                                                        max_features=rf_max_f,
                                                        min_samples_leaf=rf_min_l,
                                                        criterion=rf_c,
                                                        max_depth=rf_max_d,
                                                        min_samples_split=rf_min_ss)

                            rf.fit(X_train, y_train)
                            valid_score = rf.score(X_valid, y_valid)

                            print(valid_score)

                            if rf_best_score < valid_score:
                                rf_best_score               = valid_score
                                rf_best_n_estimators        = rf_n
                                rf_best_max_features        = rf_max_f
                                rf_best_min_samples_leaf    = rf_min_l
                                rf_best_criterion           = rf_c
                                rf_best_max_depth           = rf_max_d
                                rf_best_min_sample_split    = rf_min_ss


    return rf_best_score, rf_best_n_estimators, rf_best_max_features, rf_best_min_samples_leaf, rf_best_criterion, rf_best_max_depth, rf_best_min_sample_split

# ------------------------------------------------------------------------------------
def rf_train_k_fold(X_labelled, y_labelled):
    rf_best_criterion = 'gini'
    rf_best_max_depth = None
    rf_best_min_sample_split = 2
    rf_best_n_estimators = 'entropy'
    rf_best_max_features = None
    rf_best_min_samples_leaf = 1
    rf_best_score = 0

    for rf_n in Config.CFG_RF_N_ESTIMATORS:
        for rf_max_f in Config.CFG_RF_MAX_FEATURES:
            for rf_min_l in Config.CFG_RF_MIN_LEAF:
                for rf_c in Config.CFG_RF_CRITERION:
                    for rf_max_d in Config.CFG_RF_MAX_DEPTH:
                        for rf_min_ss in Config.CFG_RF_MIN_SPLIT:
                            rf = RandomForestClassifier(n_estimators=rf_n,
                                                        max_features=rf_max_f,
                                                        min_samples_leaf=rf_min_l,
                                                        criterion=rf_c,
                                                        max_depth=rf_max_d,
                                                        min_samples_split=rf_min_ss)

                            # begin k-fold loop
                            k_fold_ranges = KFold(n_splits=Config.CFG_RF_K_FOLD)
                            k_fold_splits = k_fold_ranges.get_n_splits(X_labelled)
                            itr_acc = 0
                            for train_index, test_index in k_fold_ranges.split(X_labelled):
                                X_train, X_valid = X_labelled[train_index], X_labelled[test_index]
                                y_train, y_valid = y_labelled[train_index], y_labelled[test_index]

                                # train the model with the k-fold training data
                                rf.fit(X_train, y_train)

                                itr_acc += rf.score(X_valid, y_valid)

                            itr_acc = itr_acc / k_fold_splits

                            if rf_best_score < itr_acc:
                                rf_best_score = itr_acc
                                rf_best_n_estimators = rf_n
                                rf_best_max_features = rf_max_f
                                rf_best_min_samples_leaf = rf_min_l
                                rf_best_criterion = rf_c
                                rf_best_max_depth = rf_max_d
                                rf_best_min_sample_split = rf_min_ss


    return rf_best_score, rf_best_n_estimators, rf_best_max_features, rf_best_min_samples_leaf, rf_best_criterion, rf_best_max_depth, rf_best_min_sample_split

# ------------------------------------------------------------------------------------
def rf_predict(best_n_estimators,
               best_max_features,
               best_min_samples_leaf,
               best_criterion,
               best_max_depth,
               best_min_samples_split,
               X_labelled, y_labelled, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=best_n_estimators,
                                max_features=best_max_features,
                                min_samples_leaf=best_min_samples_leaf,
                                criterion=best_criterion,
                                max_depth=best_max_depth,
                                min_samples_split=best_min_samples_split)
    rf.fit(X_labelled, y_labelled)

    # compute the performance metrics
    y_predict       = rf.predict(X_test)
    confusion_mc    = confusion_matrix(y_test, y_predict)
    metrics_report  = classification_report(y_test, y_predict)

    # get the validation accuracy of the KNN algorithm
    return metrics.accuracy_score(y_test, y_predict), confusion_mc, metrics_report

# ------------------------------------------------------------------------------------
def rf_main(X_train_dec, y_train_dec, X_valid_dec, y_valid_dec,
            X_train_bin, y_train_bin, X_valid_bin, y_valid_bin,
            X_labelled_dec, y_labelled_dec, X_labelled_bin, y_labelled_bin,
            X_test_dec, y_test_dec, X_test_bin, y_test_bin,
            rf_validation_mode):

    if rf_validation_mode == Config.ValidationMethod.CROSS_VALIDATION:
        print('\n[4-1] Random Forest Classification - Cross Validation 3:1:1')
        print('====================================================================================')

        # get Random Forest algorithm training results
        rf_train_score_dec, \
        rf_best_n_estimators_dec, \
        rf_best_max_features_dec, \
        rf_best_min_leaf_dec, \
        rf_best_criterion_dec, \
        rf_best_max_depth_dec, \
        rf_best_min_samples_split_dec = rf_train_cross_validate(X_train_dec, y_train_dec, X_valid_dec, y_valid_dec)

        rf_train_score_bin, \
        rf_best_n_estimators_bin, \
        rf_best_max_features_bin, \
        rf_best_min_leaf_bin, \
        rf_best_criterion_bin, \
        rf_best_max_depth_bin, \
        rf_best_min_samples_split_bin = rf_train_cross_validate(X_train_bin, y_train_bin, X_valid_bin, y_valid_bin)

        # get Random Forest algorithm testing results
        rf_test_score_dec, rf_test_cm_dec, rf_test_mt_dec = rf_predict(rf_best_n_estimators_dec,
                                                                       rf_best_max_features_dec,
                                                                       rf_best_min_leaf_dec,
                                                                       rf_best_criterion_dec,
                                                                       rf_best_max_depth_dec,
                                                                       rf_best_min_samples_split_dec,
                                                                       X_train_dec,y_train_dec, X_test_dec, y_test_dec)

        rf_test_score_bin, rf_test_cm_bin, rf_test_mt_bin = rf_predict(rf_best_n_estimators_bin,
                                                                       rf_best_max_features_bin,
                                                                       rf_best_min_leaf_bin,
                                                                       rf_best_criterion_bin,
                                                                       rf_best_max_depth_bin,
                                                                       rf_best_min_samples_split_bin,
                                                                       X_train_bin, y_train_bin, X_test_bin, y_test_bin)
    else:
        print('\n[4-2] Random Forest Classification - k-fold Validation (k=', Config.CFG_RF_K_FOLD, ')')
        print('====================================================================================')

        # get Random Forest algorithm training results
        rf_train_score_dec, \
        rf_best_n_estimators_dec, \
        rf_best_max_features_dec, \
        rf_best_min_leaf_dec, \
        rf_best_criterion_dec, \
        rf_best_max_depth_dec, \
        rf_best_min_samples_split_dec = rf_train_k_fold(X_labelled_dec, y_labelled_dec)

        rf_train_score_bin, \
        rf_best_n_estimators_bin, \
        rf_best_max_features_bin, \
        rf_best_min_leaf_bin, \
        rf_best_criterion_bin, \
        rf_best_max_depth_bin, \
        rf_best_min_samples_split_bin = rf_train_k_fold(X_labelled_bin, y_labelled_bin)

        # get Random Forest algorithm testing results
        rf_test_score_dec, rf_test_cm_dec, rf_test_mt_dec = rf_predict(rf_best_n_estimators_dec,
                                                                       rf_best_max_features_dec,
                                                                       rf_best_min_leaf_dec,
                                                                       rf_best_criterion_dec,
                                                                       rf_best_max_depth_dec,
                                                                       rf_best_min_samples_split_dec,
                                                                       X_labelled_dec, y_labelled_dec, X_test_dec, y_test_dec)

        rf_test_score_bin, rf_test_cm_bin, rf_test_mt_bin = rf_predict(rf_best_n_estimators_bin,
                                                                       rf_best_max_features_bin,
                                                                       rf_best_min_leaf_bin,
                                                                       rf_best_criterion_bin,
                                                                       rf_best_max_depth_bin,
                                                                       rf_best_min_samples_split_bin,
                                                                       X_labelled_bin, y_labelled_bin, X_test_bin, y_test_bin)

    # printing the validation score of the Random Forest training process
    print('=> Decimal Features: (Random Forest Validation Score: ', rf_train_score_dec, ')')
    print('=> Binary Features:  (Random Forest Validation Score: ', rf_train_score_bin, ')')

    print('=> Decimal Features: (Random Forest Testing Score: \t', rf_test_score_dec, ')')
    print('=> Binary Features:  (Random Forest Testing Score: \t', rf_test_score_bin, ')')