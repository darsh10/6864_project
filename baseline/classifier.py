from baseline import *
import random
from sklearn import svm
from sklearn import linear_model
import pickle
import os
from util import element_wise_dot
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

class ClassifierBaseline(Baseline):
    def __init__(self, use_nn=False, data_size=sys.maxint, download_support_docs=False):
        super(ClassifierBaseline, self).__init__(data_size=data_size, download_support_docs=download_support_docs)
        self.use_nn = use_nn
        self.use_binary_bow = not self.use_nn

    def do_classifier_baseline(self):
        map_support_docs_bow = {}
        f_name = "map_support_docs_bow_%d_%sbinary.pkl"%(len(self.solved_files), "" if self.use_binary_bow else "non")
        print "\n\nComputing support doc BoW vectors"
        if os.path.isfile(f_name):
            with open(f_name, "r") as f:
                map_support_docs_bow = pickle.load(f)
        if not map_support_docs_bow:
            for doc, super_title in self.map_support_docs_super_titles.items():
                content_text = self.map_support_docs_text[doc]
                total_text = super_title+" "+content_text
                map_support_docs_bow[doc] = bow_vector(total_text.split(' '), self.all_words, binary=self.use_binary_bow)
            with open(f_name, "w") as f:
                pickle.dump(map_support_docs_bow, f)

        map_file_bow = {}
        f_name = "map_file_bow_%d_%sbinary.pkl"%(len(self.solved_files), "" if self.use_binary_bow else "non")
        print "\n\nComputing query doc BoW vectors"
        if os.path.isfile(f_name):
            with open(f_name, "r") as f:
                map_file_bow = pickle.load(f)
        if not map_file_bow:
            for i, text in enumerate(self.solved_files_question_text):
                file_bow=bow_vector(text.split(' '), self.all_words, binary=self.use_binary_bow)
                map_file_bow[self.solved_files[i]] = file_bow
            with open(f_name, "w") as f:
                pickle.dump(map_file_bow, f)

        print "\n\nConstructing training data"
        map_support_doc_pos_train_count = {}
        positive_data_x = []
        positive_data_y = []
        train_test_index = int(len(self.solved_files)*0.8)
        for file_name in self.solved_files[:train_test_index]:
            cor_doc = self.solved_files_support_docs[file_name]
            if self.use_nn:
                positive_data_x.append(np.concatenate((map_file_bow[file_name], map_support_docs_bow[cor_doc])))
            else:
                positive_data_x.append(element_wise_dot(map_file_bow[file_name], map_support_docs_bow[cor_doc]))
            map_support_doc_pos_train_count[cor_doc] = map_support_doc_pos_train_count.setdefault(cor_doc,0) + 1
            positive_data_y.append(1.0)
            if len(positive_data_y) % 1000 == 0:
                print "Collected %d positive data points"%len(positive_data_y)

        map_support_doc_neg_train_count = {}
        random.seed(100)
        negative_data_x = []
        negative_data_y = []
        for file_name in self.solved_files[:train_test_index]:
            cor_doc = self.solved_files_support_docs[file_name]
            cor_ind = self.all_support_docs.index(cor_doc)
            start_ind = 0
            end_ind = len(self.all_support_docs)-1
            for j in range(3):
                cur_ind = random.randint(start_ind, end_ind)
                if cur_ind==cor_ind:
                    cur_ind = (cur_ind+1)%end_ind
                cur_doc = self.all_support_docs[cur_ind]
                if self.use_nn:
                    negative_data_x.append(np.concatenate((map_file_bow[file_name], map_support_docs_bow[cur_doc])))
                else:
                    negative_data_x.append(element_wise_dot(map_file_bow[file_name], map_support_docs_bow[cur_doc]))
                map_support_doc_neg_train_count[cur_doc] = map_support_doc_neg_train_count.setdefault(cur_doc,0) + 1
                negative_data_y.append(0.0)
                if len(negative_data_y) % 1000 == 0:
                    print "Collected %d negative data points"%len(negative_data_y)
        for support_doc in self.all_support_docs:
            for j in range(2):
                cur_ind = random.randint(0, train_test_index-1)
                query_doc = self.solved_files[cur_ind]
                while self.solved_files_support_docs[query_doc] == support_doc:
                    cur_ind = random.randint(0, train_test_index-1)
                    query_doc = self.solved_files[cur_ind]
                if self.use_nn:
                    negative_data_x.append(np.concatenate((map_file_bow[query_doc], map_support_docs_bow[support_doc])))
                else:
                    negative_data_x.append(element_wise_dot(map_file_bow[query_doc], map_support_docs_bow[support_doc]))
                map_support_doc_neg_train_count[support_doc] = map_support_doc_neg_train_count.setdefault(support_doc,0) + 1
                negative_data_y.append(0.0)
                if len(negative_data_y) % 1000 == 0:
                    print "Collected %d negative data points"%len(negative_data_y)

        if self.use_nn:
            model = Sequential()
            model.add(Dense(input_dim=2*len(self.all_words),output_dim=len(self.all_words),activation='sigmoid'))
            model.add(Dense(input_dim=len(self.all_words),output_dim=1,activation='sigmoid'))
            sgd = SGD()
            model.compile(loss='mean_squared_error',optimizer=sgd)
        else:
            logreg = linear_model.LogisticRegression(C=1e3)
        print "Positive training counts for support documents:"
        print sorted(map_support_doc_pos_train_count.items(), key=lambda x: -x[1])
        print "Negative training counts for support documents:"
        print sorted(map_support_doc_neg_train_count.items(), key=lambda x: -x[1])
        print "\n\nTraining\n"
        data_x = np.array(positive_data_x + negative_data_x)
        if self.use_nn:
            assert data_x.shape[1] == 2 * len(self.all_words)
        else:
            assert data_x.shape[1] == len(self.all_words)
        data_y = np.array(positive_data_y + negative_data_y)
        assert data_x.shape[0] == data_y.shape[0]
        if self.use_nn:
            model.fit(data_x, data_y, nb_epoch=100, batch_size=data_y.shape[0]/10, verbose=2)
        else:
            logreg.fit(data_x, data_y)
        del positive_data_x, positive_data_y, negative_data_x, negative_data_y, data_x, data_y

        print "\n\nDetermining training accuracy\n"
        correct = 0
        total_considered = 0
        for file_name in self.solved_files[:train_test_index]:
            cor_doc = self.solved_files_support_docs[file_name]
            file_bow = map_file_bow[file_name]
            max_score = 0.0
            max_score_doc = ""
            all_scores = {}
            for cur_doc in self.all_support_docs:
                cur_bow = map_support_docs_bow[cur_doc]
                if self.use_nn:
                    test_data = np.concatenate((file_bow, cur_bow)).reshape(1,2*len(self.all_words))
                    cur_score = model.predict_proba(test_data,verbose=0)[0][0]
                else:
                    cur_score = logreg.predict_proba([element_wise_dot(file_bow, cur_bow)])[0][1]
                all_scores[cur_doc] = cur_score
                if cur_score>max_score:
                    max_score = cur_score
                    max_score_doc = cur_doc
            if max_score_doc==cor_doc:
                correct += 1
            total_considered += 1
            if total_considered%10 == 0:
                print "\nSo far got %d correct of %d" %(correct,total_considered)
            print "Top 10 scoring documents:", sorted(all_scores.items(), key=lambda x: -x[1])[:10]
            print "Pred: %s"%max_score_doc
            print "Actual: %s"%cor_doc

        train_accuracy = correct*1.0/total_considered
        print "Correct %d of %d" %(correct,total_considered)
        print "Classifier training accuracy of %f" %train_accuracy

        print "\n\nDetermining testing accuracy\n"
        correct = 0
        total_considered = 0
        for file_name in self.solved_files[train_test_index:]:
            cor_doc = self.solved_files_support_docs[file_name]
            file_bow = map_file_bow[file_name]
            max_score = 0.0
            max_score_doc = ""
            all_scores = {}
            for cur_doc in self.all_support_docs:
                cur_bow = map_support_docs_bow[cur_doc]
                if self.use_nn:
                    test_data = np.concatenate((file_bow, cur_bow)).reshape(1,2*len(self.all_words))
                    cur_score = model.predict_proba(test_data,verbose=0)[0][0]
                else:
                    cur_score = logreg.predict_proba([element_wise_dot(file_bow, cur_bow)])[0][1]
                all_scores[cur_doc] = cur_score
                if cur_score>max_score:
                    max_score = cur_score
                    max_score_doc = cur_doc
            if max_score_doc==cor_doc:
                correct += 1
            total_considered += 1
            if total_considered%10 == 0:
                print "\nSo far got %d correct of %d" %(correct,total_considered)
            print "Top 10 scoring documents:", sorted(all_scores.items(), key=lambda x: -x[1])[:10]
            print "Pred: %s"%max_score_doc
            print "Actual: %s"%cor_doc

        test_accuracy = correct*1.0/total_considered
        print "\n\nCorrect %d of %d" %(correct,total_considered)
        print "Classifier testing accuracy of %f" %test_accuracy

        print "\n\nTraining accuracy: %f"%train_accuracy
        print "Testing accuracy: %f" %test_accuracy
        return test_accuracy

def main():
    usage = "python classifier.py [-n <data size>] [-k to use neural network, default: logistic regression] [-d to download missing support documents]"
    data_size = sys.maxint
    download_support_docs = False
    use_nn = False
    top_n = 1

    try:
        opts, args = getopt.getopt(sys.argv[1:], "kn:d")
    except getopt.GetoptError:
        print usage
        sys.exit(1)
    for opt, arg in opts:
        if opt == "-n":
            data_size = int(arg)
        elif opt == "-d":
            download_support_docs=True
        elif opt == "-k":
            use_nn = True

    if data_size < sys.maxint:
        print "Only using %d samples."%data_size
    else:
        print "Using full dataset."
    if download_support_docs:
        print "Will crawl missing support documents."
    else:
        print "Will not crawl missing support documents."
    if use_nn:
        print "Using neural network."
    else:
        print "Using logistic regression."
    print
    b = ClassifierBaseline(use_nn=use_nn, data_size=data_size, download_support_docs=download_support_docs)
    b.populate_data()
    b.do_classifier_baseline()

if __name__ == "__main__":
    main()
