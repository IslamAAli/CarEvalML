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

import numpy as np
import pandas as pd

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

    # get baseline values (majority guess)
    baselineMajorityGuess.baseline_majority_guess(ds_decimal_processed)

    # ========================================================================================================

    # KNN classification with cross validation 3:1:1
    KNNModel.knn_main_cross_validate(X_train_dec, y_train_dec, X_valid_dec, y_valid_dec,
                                     X_train_bin, y_train_bin, X_valid_bin, y_valid_bin,
                                     X_test_dec, y_test_dec, X_test_bin, y_test_bin)

    # KNN classification with k-fold validation
    KNNModel.knn_main_k_fold(X_labelled_dec, y_labelled_dec, X_labelled_bin, y_labelled_bin,
                            X_test_dec, y_test_dec, X_test_bin, y_test_bin)

    # ========================================================================================================

    # Naive Bayes classification with cross validation 3:1:1
    NaiveBayesModel.naive_bayes_main_cross_validate(X_train_dec, y_train_dec, X_valid_dec, y_valid_dec,
                                                    X_train_bin, y_train_bin, X_valid_bin, y_valid_bin,
                                                    X_test_dec, y_test_dec, X_test_bin, y_test_bin)

    # Naive Bayes classification with k-fold validation
    NaiveBayesModel.naive_bayes_main_k_fold(X_labelled_dec, y_labelled_dec, X_labelled_bin, y_labelled_bin,
                                            X_test_dec, y_test_dec, X_test_bin, y_test_bin)

    # ========================================================================================================

if __name__ == "__main__":
    main()