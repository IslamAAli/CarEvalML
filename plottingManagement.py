import Config
import seaborn as sns
import matplotlib.pyplot as plt
from terminaltables import AsciiTable


def plot_confusion_matrix(m_conf_mat, title):
    ax = plt.subplot()
    sns.heatmap(m_conf_mat, annot=True, ax=ax);

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    # ax.xaxis.set_ticklabels(['business', 'health']);
    # ax.yaxis.set_ticklabels(['health', 'business']);
    plt.show()

def plot_summary_table():
    table_data = [
        ['ID', 'Algorithm', 'Validation Score (dec)', 'Testing Score (dec)', 'Validation Score (bin)', 'Testing Score (bin)'],
        ['0', 'Majority Vote (Baseline)', 1.0, Config.RES_BASE_LINE, 1.0, Config.RES_BASE_LINE],
        ['1', 'KNN', Config.RES_VAL_KNN_DEC, Config.RES_TEST_KNN_DEC, Config.RES_VAL_KNN_BIN, Config.RES_TEST_KNN_BIN],
        ['2', 'Naive Bayes', Config.RES_VAL_NAIVE_BAYES_DEC, Config.RES_TEST_NAIVE_BAYES_DEC, Config.RES_VAL_NAIVE_BAYES_BIN, Config.RES_TEST_NAIVE_BAYES_BIN],
        ['3', 'Decision Tree', Config.RES_VAL_DEC_TREE_DEC, Config.RES_TEST_DEC_TREE_DEC, Config.RES_VAL_DEC_TREE_BIN, Config.RES_TEST_DEC_TREE_BIN],
        ['4', 'Random Forest', Config.RES_VAL_RAND_FOREST_DEC, Config.RES_TEST_RAND_FOREST_DEC, Config.RES_VAL_RAND_FOREST_BIN, Config.RES_TEST_RAND_FOREST_BIN],
        ['5', 'SVM', Config.RES_VAL_SVM_DEC, Config.RES_TEST_SVM_DEC, Config.RES_VAL_SVM_BIN, Config.RES_TEST_SVM_BIN],
        ['6', 'CNN', 0, 0, 0, 0],
    ]
    table = AsciiTable(table_data)
    print(table.table)