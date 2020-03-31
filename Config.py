from enum import Enum
import numpy as np

CFG_debug       = 0
CFG_KNN_max_n   = 30
CFG_KNN_K_FOLD  = 5

CFG_NVB_ALPHA_STEP  = 0.1
CFG_NVB_K_FOLD      = 5

CFG_DT_CRITERION = ['gini', 'entropy']
CFG_DT_MAX_DEPTH = [None, 1, 5, 10, 30, 50, 100, 1000]
CFG_DT_MIN_SPLIT = np.arange(2,10)
CFG_DT_MIN_LEAF  = np.arange(1,10)
CFG_DT_K_FOLD    = 5


class ValidationMethod(Enum):
    CROSS_VALIDATION    = 1
    K_FOLD_VALIDATION   = 2