from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix,classification_report
from sklearn import metrics

import Config
import plottingManagement

# ------------------------------------------------------------------------------------
def naive_bayes_train_cross_validate(X_train, y_train, X_valid, y_valid):
    naive_bayes_best_alpha = 0
    naive_bayes_best_score = 0

    m_alpha = 0
    for i in range(int(1/Config.CFG_NVB_ALPHA_STEP)):
        m_alpha = m_alpha + Config.CFG_NVB_ALPHA_STEP
        naive_bayes = MultinomialNB(alpha=m_alpha)
        naive_bayes.fit(X_train, y_train)

        if Config.CFG_debug == 1:
            print("\nn= ", i + 1, " - naive_bayes Score = ", naive_bayes.score(X_valid, y_valid))

        validation_score = naive_bayes.score(X_valid, y_valid)

        if naive_bayes_best_score < validation_score:
            naive_bayes_best_score = validation_score
            naive_bayes_best_alpha = m_alpha

    return naive_bayes_best_score, naive_bayes_best_alpha

# ------------------------------------------------------------------------------------
def naive_bayes_train_k_fold(X_labelled, y_labelled):
    naive_bayes_best_alpha = 0
    naive_bayes_best_score = 0

    m_alpha = 0
    for i in range(Config.CFG_KNN_max_n):
        m_alpha = m_alpha + Config.CFG_NVB_ALPHA_STEP
        naive_bayes = MultinomialNB(alpha=m_alpha)

        # begin k-fold loop
        nvb_ranges = KFold(n_splits=Config.CFG_NVB_K_FOLD)
        k_fold_splits = nvb_ranges.get_n_splits(X_labelled)
        itr_acc = 0
        for train_index, test_index in nvb_ranges.split(X_labelled):
            X_train, X_valid = X_labelled[train_index], X_labelled[test_index]
            y_train, y_valid = y_labelled[train_index], y_labelled[test_index]

            # train the model with the k-fold training data
            naive_bayes.fit(X_train, y_train)

            if Config.CFG_debug == 1:
                print('\nn= ', i + 1, " - KNN Score = ", naive_bayes.score(X_labelled, y_labelled))

            itr_acc += naive_bayes.score(X_valid, y_valid)

        itr_acc = itr_acc / k_fold_splits

        if naive_bayes_best_score < itr_acc:
            naive_bayes_best_score = naive_bayes.score(X_valid, y_valid)
            naive_bayes_best_alpha = m_alpha

    return naive_bayes_best_score, naive_bayes_best_alpha

# ------------------------------------------------------------------------------------
def naive_bayes_predict(nvb_best_alpha, X_labelled, y_labelled, X_test, y_test):
    naive_bayes = MultinomialNB(alpha=nvb_best_alpha)
    naive_bayes.fit(X_labelled, y_labelled)

    # compute the performance metrics
    y_predict = naive_bayes.predict(X_test)
    confusion_mc = confusion_matrix(y_test, y_predict)
    metrics_report = classification_report(y_test, y_predict, zero_division=1)

    # get the validation accuracy of the naive_bayes algorithm
    return metrics.accuracy_score(y_test, y_predict), confusion_mc, metrics_report

# ------------------------------------------------------------------------------------
def naive_bayes_main_cross_validate(X_train_dec, y_train_dec, X_valid_dec, y_valid_dec,
                                    X_train_bin, y_train_bin, X_valid_bin, y_valid_bin,
                                    X_test_dec, y_test_dec, X_test_bin, y_test_bin):
    print('\n[2-1] Naive Bayes Classification - Cross Validation 3:1:1')
    print('====================================================================================')

    # get naive_bayes algorithm training results
    nvb_train_score_dec, nvb_best_alpha_dec = naive_bayes_train_cross_validate(X_train_dec, y_train_dec, X_valid_dec, y_valid_dec)
    nvb_train_score_bin, nvb_best_alpha_bin = naive_bayes_train_cross_validate(X_train_bin, y_train_bin, X_valid_bin, y_valid_bin)

    # printing the validation score of the naive_bayes training process
    print('=> Decimal Features: (Naive Bayes Validation Score: ', nvb_train_score_dec, ')')
    print('=> Binary Features:  (Naive Bayes Validation Score: ', nvb_train_score_bin, ')')

    # get naive_bayes algorithm testing results
    nvb_test_score_dec, nvb_test_cm_dec, nvb_test_mt_dec = naive_bayes_predict(nvb_best_alpha_dec, X_train_dec, y_train_dec, X_test_dec, y_test_dec)
    nvb_test_score_bin, nvb_test_cm_bin, nvb_test_mt_bin = naive_bayes_predict(nvb_best_alpha_bin, X_train_bin, y_train_bin, X_test_bin, y_test_bin)

    print('=> Decimal Features: (Naive Bayes Testing Score: \t', nvb_test_score_dec, ')')
    print('=> Binary Features:  (Naive Bayes Testing Score: \t', nvb_test_score_bin, ')')


# ------------------------------------------------------------------------------------
def naive_bayes_main_k_fold(X_labelled_dec, y_labelled_dec, X_labelled_bin, y_labelled_bin,
                            X_test_dec, y_test_dec, X_test_bin, y_test_bin):

    print('\n[2-2] Naive Bayes Classification - k-fold Validation (k=', Config.CFG_NVB_K_FOLD, ')')
    print('====================================================================================')

    # get naive_bayes algorithm training results
    nvb_train_score_dec, nvb_best_alpha_dec = naive_bayes_train_k_fold(X_labelled_dec, y_labelled_dec)
    nvb_train_score_bin, nvb_best_alpha_bin = naive_bayes_train_k_fold(X_labelled_bin, y_labelled_bin)

    # printing the validation score of the naive_bayes training process
    print('=> Decimal Features: (Naive Bayes Validation Score: ', nvb_train_score_dec, ')')
    print('=> Binary Features:  (Naive Bayes Validation Score: ', nvb_train_score_bin, ')')

    # get naive_bayes algorithm testing results
    nvb_test_score_dec, nvb_test_cm_dec, nvb_test_mt_dec = naive_bayes_predict(nvb_best_alpha_dec, X_labelled_dec, y_labelled_dec, X_test_dec, y_test_dec)
    nvb_test_score_bin, nvb_test_cm_bin, nvb_test_mt_bin = naive_bayes_predict(nvb_best_alpha_bin, X_labelled_bin, y_labelled_bin, X_test_bin, y_test_bin)

    print('=> Decimal Features: (Naive Bayes Testing Score: \t', nvb_test_score_dec, ')')
    print('=> Binary Features:  (Naive Bayes Testing Score: \t', nvb_test_score_bin, ')')