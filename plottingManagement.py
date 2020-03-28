import Config
import seaborn as sns
import matplotlib.pyplot as plt

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