import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# read the CSV file, mainly it is the same as the data file but with added header
def dataset_read_unprocessed():
    ds_unprocessed = pd.read_csv('Dataset/car.data.csv')
    return ds_unprocessed

# replace words in dataset with decimal numerical values for classification
def dataset_process_decimal(m_ds_unprocessed):
    ds_decimal_processed = m_ds_unprocessed

    # mapping buying field
    buying_dict = [ ['vhigh', 5], ['high', 4], ['med', 3], ['low', 1] ]
    for i in range(len(buying_dict)):
        ds_decimal_processed.buying[ds_decimal_processed.buying == buying_dict[i][0]] = buying_dict[i][1]

    # mapping the maint field
    maint_dict = [['vhigh', 5], ['high', 4], ['med', 3], ['low', 1]]
    for i in range(len(maint_dict)):
        ds_decimal_processed.maint[ds_decimal_processed.maint == maint_dict[i][0]] = maint_dict[i][1]

    # mapping the door field (only one field needs conversion)
    ds_decimal_processed.doors[ds_decimal_processed.doors == '5more'] = 5

    # mapping the persons field (only one field needs conversion)
    ds_decimal_processed.persons[ds_decimal_processed.persons == 'more'] = 5

    # mapping the lug_boot field
    lug_boot_dict = [['small', 1], ['med', 3], ['big', 5]]
    for i in range(len(lug_boot_dict)):
        ds_decimal_processed.lug_boot[ds_decimal_processed.lug_boot == lug_boot_dict[i][0]] = lug_boot_dict[i][1]

    # mapping the safety field
    safety_dict = [['low', 1], ['med', 3], ['high', 5]]
    for i in range(len(lug_boot_dict)):
        ds_decimal_processed.safety[ds_decimal_processed.safety == safety_dict[i][0]] = safety_dict[i][1]

    X_data = ds_decimal_processed.loc[:, 'buying':'safety']
    y_data = ds_decimal_processed.loc[:, 'car_class']

    X_data, y_data = X_data.values.astype(int) , y_data.values.astype(int)

    return X_data, y_data, ds_decimal_processed

# convert categoral data to binary integer data to be valid input/output for the ML algorithms
def dataset_process_binary(m_ds_unprocessed):
    X_data = pd.get_dummies(m_ds_unprocessed.loc[:,'buying':'safety'])
    y_data = m_ds_unprocessed.loc[:, 'car_class']

    # mapping class field
    car_class_dict = [['unacc', 1], ['acc', 2], ['good', 3], ['vgood', 4]]
    for i in range(len(car_class_dict)):
        y_data[y_data== car_class_dict[i][0]] = car_class_dict[i][1]

    X_data, y_data = X_data.values.astype(int), y_data.values.astype(int)

    return X_data, y_data

# split the data into training and testing data based on certain test data size
def split_data_train_test(m_X, m_y, m_test_size):
    X_train, X_test, y_train, y_test = train_test_split(m_X, m_y, test_size=m_test_size, random_state=42)

    return X_train, X_test, y_train, y_test
