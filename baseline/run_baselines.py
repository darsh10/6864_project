import getopt
import sys
from matplotlib import pyplot as plt
import numpy as np

from cosine_similarity import CosineSimilarityBaseline
from multi_label_classifier import ClassifierBaseline

def run_all_baselines():
    usage = "python run_baselines.py [-t to remove tf-idf] [-n <data size>]"
    data_size = sys.maxint
    try:
        opts, args = getopt.getopt(sys.argv[1:], "tn:")
    except getopt.GetoptError:
        print usage
        sys.exit(1)
    for opt, arg in opts:
        if opt == "-n":
            data_size = int(arg)
    if data_size < sys.maxint:
        print "Only using %d samples."%data_size
    else:
        print "Using full dataset."

    print "\n\n\nDOING COSINE SIM BASELINE WITH TF-IDF\n\n\n"
    cos_sim_b = CosineSimilarityBaseline(do_tf_idf=True, data_size=data_size, download_support_docs=False)
    cos_sim_b.populate_data()
    tfidf_score = cos_sim_b.do_cosine_similarity()
    cos_sim_b.do_tf_idf = False
    print "\n\n\nDOING COSINE SIM BASELINE NO TF-IDF\n\n\n"
    bow_score = cos_sim_b.do_cosine_similarity()

    print "\n\n\nDOING MULTI LABEL CLASSIFIER BASELINE\n\n\n"
    class_b = ClassifierBaseline(data_size=data_size)
    class_b.populate_data()
    clfs, scores = class_b.run_classifiers()
        
    clfs.append('Cos Sim (BoW)')
    clfs.append('Cos Sim (TfIdf)')

    scores.append(bow_score)
    scores.append(tfidf_score)

    if data_size < sys.maxint:
        plot_scores(scores, clfs, 'all_scores_%d.png'%data_size)
    else:
        plot_scores(scores, clfs, 'all_scores_full_data.png')

    print 'Plotted model accuracies using %d samples'%data_size

def plot_double_scores(scores, scores2, labels, output_f_name):
    N = len(scores)
    ind = np.arange(N)
    width = 0.35

    fig, ax = plt.subplots()
    fig.tight_layout()

    rects1 = ax.bar(ind, scores, width, color='#73bfe5', edgecolor="none")
    rects2 = ax.bar(ind + width, scores2, width, color='#0d254c', edgecolor="none")

    ax.set_ylabel('Scores (5-fold cross validation)')
    ax.set_title('Baseline model accuracies')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(labels)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.legend((rects1[0], rects2[0]), ('2.3k questions', '5.6k questions'), frameon=False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.savefig(output_f_name, bbox_inches='tight')

def plot_scores(scores, labels, output_f_name):
    N = len(scores)
    ind = np.arange(N)
    width = 0.65

    fig, ax = plt.subplots()
    fig.tight_layout()

    rects1 = ax.bar(ind, scores, width, color='#73bfe5', edgecolor="none")

    ax.set_ylabel('Scores (5-fold cross validation)')
    ax.set_title('Baseline model accuracies')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(labels)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.savefig(output_f_name, bbox_inches='tight')

def hard_code_plots():
    scores = [0.04, 0.14, 0.24, 0.076, 0.09]
    clf = ['SVM', 'DecisionTree', 'LogisticRegression', 'Cos Sim (BoW)', 'Cos Sim (TfIdf)']

    plot_scores(scores, clf, 'all_scores_93159.png')

    scores2 = [0.04, 0.1562, 0.2832, 0.063, 0.065]
    clf = ['SVM', 'DecisionTree', 'LogisticRegression', 'Cos Sim (BoW)', 'Cos Sim (TfIdf)']
    # plot_scores(scores, clf, 'all_scores_full_data.png')

    plot_double_scores(scores, scores2, clf, 'scores_partial_full.png')

if __name__ == '__main__':
    run_all_baselines()
