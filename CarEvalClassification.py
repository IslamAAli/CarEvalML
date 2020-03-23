## --------------------------------------------------------------------------
## Car Evaluation Classification - CMPUT 566 Course Project
## Developed by: Islam A. Ali
## Department of Computing Science - University of Alberta
## --------------------------------------------------------------------------

import Config
import datasetManagment
import baselineMajorityGuess
import numpy as np
import pandas as pd

def main():
    # reading the data set
    ds_unprocessed = datasetManagment.dataset_read_unprocessed()
    if Config.CFG_debug == 1:
        print(ds_unprocessed.head())
        print('=====================')

    X_data_binary, y_data_binary = datasetManagment.dataset_process_binary(ds_unprocessed)
    if Config.CFG_debug == 1:
        print(X_data_binary.head(), '\n', y_data_binary.head())
        print('=====================')

    # convert string values to integers for classification to work
    X_data_decimal, y_data_decimal, ds_decimal_processed = datasetManagment.dataset_process_decimal(ds_unprocessed)
    if Config.CFG_debug == 1:
        print(X_data_decimal.head(), '\n', y_data_decimal.head())
        print('=====================')

    # get baseline values (majority guess)
    baselineMajorityGuess.baseline_majority_guess(ds_decimal_processed)



if __name__ == "__main__":
    main()