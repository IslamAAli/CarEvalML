from enum import Enum


CFG_debug       = 0
CFG_KNN_max_n   = 30
CFG_KNN_K_FOLD  = 5

CFG_NVB_ALPHA_STEP  = 0.1
CFG_NVB_K_FOLD      = 5

class ValidationMethod(Enum):
    CROSS_VALIDATION    = 1
    K_FOLD_VALIDATION   = 2