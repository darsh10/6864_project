import getopt
import numpy as np
import sys

class SVMScoreCalc(object):
    def __init__(self, scoring_n=1, p_fname=None, t_fname=None, verbose=False):
        predictions_fname = p_fname
        test_fname = t_fname

        self.scoring_n = scoring_n
        self.pred_f = open(predictions_fname)
        self.test_f = open(test_fname)

        self.verbose = verbose

        self.prev_qid = None
        self.correct = 0.
        self.total = 0.
        self.data_for_cur_qid = list()
        self.ranks_for_cur_qid = list()

    def calc_scores(self, verbose=False):
        for pred, test in zip(self.pred_f, self.test_f):
            self.cur_pred = pred
            self.cur_test = test
            pred_score = float(pred)
            test_components = test.split(' ')
            cur_qid = int(test_components[1].split(':')[1])
            if cur_qid == self.prev_qid or self.prev_qid is None:
                self.data_for_cur_qid.append(pred_score)
                self.prev_qid = cur_qid
            else:
                self.calc_score_qid()
                self.prev_qid = cur_qid
                self.data_for_cur_qid = list()
                self.data_for_cur_qid.append(pred_score)

        self.calc_score_qid()
        self.acc = self.correct / self.total
        print 'Top-%d accuracy: %.4f' % (self.scoring_n, self.acc)

    def calc_score_qid(self):
        sorted_scores = np.argsort(self.data_for_cur_qid)
        top_results = sorted_scores[-self.scoring_n:]
        irene = sorted_scores.tolist()
        if self.verbose:
            print 'top answer was ranked %d of %d' % (irene.index(0) + 1, len(sorted_scores))

        is_correct_in_top_results = 0 in top_results
        if is_correct_in_top_results:
            self.correct += 1
        self.total += 1

    def close_f(self):
        self.pred_f.close()
        self.test_f.close()

if __name__ == '__main__':
    usage = "python svm_rank_scoring.py [-t <top N>] [-v <verbose>] [-p <predictions filename] [-d test.dat filename]"
    scoring_n = 1
    verbose_bool = False
    predictions_fname = None
    test_fname = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:vp:d:")
    except getopt.GetoptError:
        print usage
        sys.exit(1)
    for opt, arg in opts:
        if opt == "-t":
            scoring_n = int(arg)
        if opt == "-v":
            verbose_bool = True
        if opt == '-p':
            predictions_fname = arg
        if opt == '-d':
            test_fname = arg

    ssc = SVMScoreCalc(scoring_n, predictions_fname, test_fname)
    ssc.calc_scores()