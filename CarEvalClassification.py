## --------------------------------------------------------------------------
## Car Evaluation Classification - CMPUT 566 Course Project
## Developed by: Islam A. Ali
## Department of Computing Science - University of Alberta
## --------------------------------------------------------------------------

import Config
import datasetManagment
import baselineMajorityGuess
import KNNModel
import numpy as np
import pandas as pd

def main():
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

    # get baseline values (majority guess)
    baselineMajorityGuess.baseline_majority_guess(ds_decimal_processed)

    # get KNN algorithm training results
    knn_train_score_dec, knn_best_n_neighbors_dec = KNNModel.knn_train(X_train_dec, y_train_dec, X_valid_dec, y_valid_dec)
    knn_train_score_bin, knn_best_n_neighbors_bin = KNNModel.knn_train(X_train_bin, y_train_bin, X_valid_bin, y_valid_bin)

    # printing the validation score of the KNN training process
    print('=> Decimal Features: (KNN Validation Score: ', knn_train_score_dec, ' - n= ', knn_best_n_neighbors_dec, ')')
    print('=> Binary Features:  (KNN Validation Score: ', knn_train_score_bin, ' - n= ', knn_best_n_neighbors_bin, ')')

    # get KNN algorithm testing results
    knn_test_score_dec = KNNModel.knn_test(knn_best_n_neighbors_dec, X_train_dec, y_train_dec, X_test_dec, y_test_dec)
    knn_test_score_bin = KNNModel.knn_test(knn_best_n_neighbors_bin, X_train_bin, y_train_bin, X_test_bin, y_test_bin)

    print('=> Decimal Features: (KNN Testing Score: ', knn_test_score_dec, ')')
    print('=> Binary Features:  (KNN Testing Score: ', knn_test_score_bin, ')')
    print('==========================================')

if __name__ == "__main__":
    main()