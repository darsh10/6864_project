"""
svm_rank.py
 - Generates train/test data files to feed into SVM Rank package. 
 - Parses output to gauge accuracy

https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html
"""
import getopt
import numpy as np
import sys
import random

from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import train_test_split

from multi_label_classifier import ClassifierBaseline

class SVMRanker(object):
    def __init__(self,data_size=sys.maxint, test_float=0.1, train_fname=None, test_fname=None):
        self.train_qid = 1
        self.test_qid = 1
        self.f = open(train_fname, 'wb')
        self.test_float = test_float

        if self.test_float > 0:
            self.f_test = open(test_fname, 'wb')

    def close_files(self):
        self.f.close()
        if self.test_float > 0:
            self.f_test.close()

    def incr_test_qid(self):
        self.test_qid += 1

    def incr_train_qid(self):
        self.train_qid += 1

    def write_row(self, q_bow, d_bow, rank, test=False):
        cur_qid = self.test_qid if test else self.train_qid
        entries = [str(rank), 'qid:%d' % cur_qid]
        for num,val in enumerate(q_bow):
            if val != 0:
                entries.append('%d:%d' % (num + 1, val))
        qlen = len(q_bow)
        for num,val in enumerate(d_bow):
            if val != 0:
                entries.append('%d:%d' % (qlen + 1 + num, val))
        if test:
            self.f_test.write(' '.join(entries))
            self.f_test.write('\n')
            self.f_test.flush()
        else:    
            self.f.write(' '.join(entries))
            self.f.write('\n')
            self.f.flush()

    def create_dat_files(self, binary=False):
        top_n_ranking = 10

        bsl = ClassifierBaseline(data_size=data_size)
        bsl.populate_data()

        support_bow = bsl._get_support_bow()
        map_file_bow = bsl._get_map_file_bow()

        X, y = bsl._get_X_and_y(map_file_bow)
        
        train_size = 500 if data_size > 5000 else None

        train_idx, test_idx = train_test_split(xrange(len(y)), train_size=train_size, random_state=42)
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        clf = lr()
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        
        train_total = 0.
        test_total = 0.
        for i in test_idx:
            test_bool = False

            if random.random() < self.test_float:
                test_bool = True
                test_total += 1
            else:
                train_total += 1

            y_actual = y[i]

            probs = clf.predict_proba(X[i])
            if test_bool:
                best_n = np.argsort(probs, axis=1)[0][::-1]
            else:
                # best_n = np.argsort(probs, axis=1)[:,-top_n_ranking:][0][::-1]
                best_n = np.argsort(probs, axis=1)[0][::-1]

            total_ranks = len(best_n)
            actual_guesses = [clf.classes_[i] for i in best_n]

            cur_doc = bsl.solved_files[i]
            cur_doc_q_bow = map_file_bow[cur_doc]
            
            # first put correct answer
            correct_support_doc = bsl.solved_files_support_docs[cur_doc]
            correct_support_bow = support_bow[correct_support_doc]

            if binary:
                self.write_row(cur_doc_q_bow, correct_support_bow, 1,test_bool)
            else:
                self.write_row(cur_doc_q_bow, correct_support_bow, total_ranks,test_bool)

            if y_actual in actual_guesses:
                actual_guesses.remove(y_actual)

            if test_bool:
                write_file_guesses = actual_guesses[:total_ranks-1]
            else:
                write_file_guesses = actual_guesses[:top_n_ranking-1]

            num_ranks = len(write_file_guesses)

            for j, guess in enumerate(write_file_guesses):
                rank = num_ranks - j
                support_doc = bsl.all_support_docs[j]
                cur_support_bow = support_bow[support_doc]
                if binary:
                    self.write_row(cur_doc_q_bow, cur_support_bow, 0,test_bool)
                else:    
                    self.write_row(cur_doc_q_bow, cur_support_bow, rank,test_bool)

            if test_bool:
                self.incr_test_qid()
            else:
                self.incr_train_qid()

        print 'Created %d train examples' % train_total
        print 'Created %d test examples' % test_total

if __name__ == '__main__':
    usage = "python svm_rank.py [-n <data size>] [-t <test float for train/test split>] [-b <binary labels>]"
    data_size = sys.maxint
    test_flt = 0.1
    binary_bool = False
    train_fname = None
    test_fname = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:t:br:s:")
    except getopt.GetoptError:
        print usage
        sys.exit(1)
    for opt, arg in opts:
        if opt == "-n":
            data_size = int(arg)
        if opt == "-t":
            test_flt = float(arg)
        if opt == "-b":
            binary_bool = True
        if opt == '-r':
            train_fname = arg
        if opt == '-s':
            test_fname = arg

    if data_size < sys.maxint:
        print "Only using %d samples." % data_size
    else:
        print "Using full dataset."

    if test_flt > 0.:
        print 'Creating qa_train.dat and qa_test.dat'

    if binary_bool:
        print 'Writing binary labels'

    svmrank = SVMRanker(data_size, test_flt, train_fname, test_fname)
    svmrank.create_dat_files(binary_bool)
    svmrank.close_files()