## --------------------------------------------------------------------------
## Car Evaluation Classification - CMPUT 566 Course Project
## Developed by: Islam A. Ali
## Department of Computing Science - University of Alberta
## --------------------------------------------------------------------------
import warnings
import Config
import datasetManagment
import baselineMajorityGuess
import KNNModel
import NaiveBayesModel
import DecisionTreeModel
import RandomForestModel
import SVMModel
import plottingManagement

def main():
    warnings.filterwarnings('once')

    # reading the data set
    ds_unprocessed = datasetManagment.dataset_read_unprocessed()
    if Config.CFG_debug == 1:
        print(ds_unprocessed.head())
        print('=====================')

    X_data_binary, y_data_binary = datasetManagment.dataset_process_binary(ds_unprocessed)

    # convert string values to integers for classification to work
    X_data_decimal, y_data_decimal, ds_decimal_processed = datasetManagment.dataset_process_decimal(ds_unprocessed)

    # split both the decimal and binary data to labeled and testing subsets
    X_labelled_bin, X_test_bin, y_labelled_bin, y_test_bin = datasetManagment.split_data_train_test(X_data_binary, y_data_binary, 0.2)
    X_labelled_dec, X_test_dec, y_labelled_dec, y_test_dec = datasetManagment.split_data_train_test(X_data_decimal, y_data_decimal, 0.2)

    # split labelled data to be for training and validation
    X_train_bin, X_valid_bin, y_train_bin, y_valid_bin = datasetManagment.split_data_train_test(X_labelled_bin, y_labelled_bin, 0.25)
    X_train_dec, X_valid_dec, y_train_dec, y_valid_dec = datasetManagment.split_data_train_test(X_labelled_dec, y_labelled_dec, 0.25)

    if Config.CFG_debug == 1:
        print(X_train_bin.shape, y_train_bin.shape, X_valid_bin.shape, y_valid_bin.shape, X_test_bin.shape, y_test_bin.shape)
        print(X_train_dec.shape, y_train_dec.shape, X_valid_dec.shape, y_valid_dec.shape, X_test_dec.shape, y_test_dec.shape)

    # ========================================================================================================

    if Config.CFG_EN_BASE_LINE ==1:
        # get baseline values (majority guess)
        baselineMajorityGuess.baseline_majority_guess(ds_decimal_processed)

    # ========================================================================================================

    if Config.CFG_EN_KNN_HOV == 1:
        # KNN classification with cross validation 3:1:1
        KNNModel.knn_main(X_train_dec, y_train_dec, X_valid_dec, y_valid_dec,
                          X_train_bin, y_train_bin, X_valid_bin, y_valid_bin,
                          X_labelled_dec, y_labelled_dec, X_labelled_bin, y_labelled_bin,
                          X_test_dec, y_test_dec, X_test_bin, y_test_bin,
                          Config.ValidationMethod.CROSS_VALIDATION)

    if Config.CFG_EN_NAIVE_BAYES_KFOLD==1:
        # KNN classification with k-fold validation
        KNNModel.knn_main(X_train_dec, y_train_dec, X_valid_dec, y_valid_dec,
                          X_train_bin, y_train_bin, X_valid_bin, y_valid_bin,
                          X_labelled_dec, y_labelled_dec, X_labelled_bin, y_labelled_bin,
                          X_test_dec, y_test_dec, X_test_bin, y_test_bin,
                          Config.ValidationMethod.K_FOLD_VALIDATION)

    # ========================================================================================================

    if Config.CFG_EN_NAIVE_BAYES_HOV==1:
        # Naive Bayes classification with cross validation 3:1:1
        NaiveBayesModel.naive_bayes_main(X_train_dec, y_train_dec, X_valid_dec, y_valid_dec,
                                         X_train_bin, y_train_bin, X_valid_bin, y_valid_bin,
                                         X_labelled_dec, y_labelled_dec, X_labelled_bin, y_labelled_bin,
                                         X_test_dec, y_test_dec, X_test_bin, y_test_bin,
                                         Config.ValidationMethod.CROSS_VALIDATION)

    if Config.CFG_EN_NAIVE_BAYES_KFOLD==1:
        # Naive Bayes classification with k-fold validation
        NaiveBayesModel.naive_bayes_main(X_train_dec, y_train_dec, X_valid_dec, y_valid_dec,
                                         X_train_bin, y_train_bin, X_valid_bin, y_valid_bin,
                                         X_labelled_dec, y_labelled_dec, X_labelled_bin, y_labelled_bin,
                                         X_test_dec, y_test_dec, X_test_bin, y_test_bin,
                                         Config.ValidationMethod.K_FOLD_VALIDATION)

    # ========================================================================================================

    if Config.CFG_EN_DEC_TREE_KFOLD==1:
        # Decision Tree classification with cross validation 3:1:1
        DecisionTreeModel.dt_main(X_labelled_dec, y_labelled_dec, X_labelled_bin, y_labelled_bin,
                                  X_test_dec, y_test_dec, X_test_bin, y_test_bin)

    # ========================================================================================================

    if Config.CFG_EN_RAND_FOREST_KFOLD == 1:
        # Decision Tree classification with k-fold validation
        RandomForestModel.rf_main(X_labelled_dec, y_labelled_dec, X_labelled_bin, y_labelled_bin,
                                  X_test_dec, y_test_dec, X_test_bin, y_test_bin)

    # ========================================================================================================

    if Config.CFG_EN_SVM_KFOLD == 1:
        # Decision Tree classification with k-fold validation
        SVMModel.svm_main(X_labelled_dec, y_labelled_dec, X_labelled_bin, y_labelled_bin,
                          X_test_dec, y_test_dec, X_test_bin, y_test_bin)

    # ========================================================================================================

    # plotting Results Summary
    plottingManagement.print_summary_table()

if __name__ == "__main__":
    main()