from enum import Enum
import numpy as np

# enable controls for different algorithms
CFG_EN_BASE_LINE            = 1
CFG_EN_KNN_HOV              = 0
CFG_EN_KNN_KFOLD            = 1
CFG_EN_NAIVE_BAYES_HOV      = 0
CFG_EN_NAIVE_BAYES_KFOLD    = 1
CFG_EN_DEC_TREE_KFOLD       = 1
CFG_EN_RAND_FOREST_KFOLD    = 1

# validation and testing results variables
RES_BASE_LINE   = 0

RES_VAL_KNN_DEC         = 0
RES_VAL_NAIVE_BAYES_DEC = 0
RES_VAL_DEC_TREE_DEC    = 0
RES_VAL_RAND_FOREST_DEC = 0
RES_VAL_KNN_BIN         = 0
RES_VAL_NAIVE_BAYES_BIN = 0
RES_VAL_DEC_TREE_BIN    = 0
RES_VAL_RAND_FOREST_BIN = 0

RES_TEST_KNN_DEC         = 0
RES_TEST_NAIVE_BAYES_DEC = 0
RES_TEST_DEC_TREE_DEC    = 0
RES_TEST_RAND_FOREST_DEC = 0
RES_TEST_KNN_BIN         = 0
RES_TEST_NAIVE_BAYES_BIN = 0
RES_TEST_DEC_TREE_BIN    = 0
RES_TEST_RAND_FOREST_BIN = 0

# debugging configuration
CFG_debug       = 0

# KNN Config.
CFG_KNN_max_n   = 30
CFG_KNN_K_FOLD  = 5

# Navie Bayes Config.
CFG_NVB_ALPHA_STEP  = 0.1
CFG_NVB_K_FOLD      = 5

# Decision Tree Config.
CFG_DT_CRITERION = ['gini', 'entropy']
CFG_DT_MAX_DEPTH = [None, 1, 5, 10, 30, 50, 100, 1000]
CFG_DT_MIN_SPLIT = np.arange(2,10)
CFG_DT_MIN_LEAF  = np.arange(1,10)
CFG_DT_K_FOLD    = 5

# Random Forest Config.
CFG_RF_CRITERION    = ['gini', 'entropy']
CFG_RF_MAX_DEPTH    = [None, 1, 5, 10, 30, 50, 100, 300, 500, 1000]
CFG_RF_N_ESTIMATORS = [10, 30, 50, 100, 500, 1000]
CFG_RF_MAX_FEATURES = ['auto', 'sqrt', 'log2', None]
CFG_RF_MIN_LEAF     = np.arange(1,10)
CFG_RF_MIN_SPLIT    = np.arange(2,10)
CFG_RF_K_FOLD       = 5


class ValidationMethod(Enum):
    CROSS_VALIDATION    = 1
    K_FOLD_VALIDATION   = 2