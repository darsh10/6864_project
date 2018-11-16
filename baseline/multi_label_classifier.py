#from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression as lr
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
#from matplotlib import pyplot as plt
import webbrowser
import collections
import pickle

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Convolution1D
#from keras.layers.pooling import GlobalMaxPooling1D
from keras.optimizers import SGD,Adagrad,Adam
from scipy.sparse import coo_matrix
from util import WordVector
from baseline import *

from collections import Counter


NUM_EPOCHS = 10
no_words = 100000
word_vector_dim = 200
Batch_Size = 32

class ClassifierBaseline(Baseline):
    def run_classifiers(self):
        print '\n\nConstructing data'
        X = []
        y = []
        if not self.run_without_word_vectors:
            X, y = self._get_X_and_y_word_vectors()
        elif self.bigram_bool:
            map_file_bigram = self._get_map_file_bigram()
            X, y = self._get_X_and_y(map_file_bigram)
        else:
            map_file_bow = self._get_map_file_bow()
            X, y = self._get_X_and_y(map_file_bow)
        clfs = ['NeuralNet']
        # clfs = ['Majority']
        scores = []
        for clf_n in clfs:
            print '\n\nDoing %s'%clf_n
            if self.do_error_analysis:
                clf_score = self._classifier_error_analysis(X, y)
            else:
                clf_score = self._classifier_score(X, y, clf_n)
            print '\n%s average score over folds: %.4f'%(clf_n, clf_score)
            scores.append(clf_score)
        return clfs, scores

    def _get_map_file_bow(self):
        map_file_bow = {}
        map_file_tfidf_bow = {}
        f_name = "map_file_bow_%d_nonbinary.pkl" % len(self.solved_files)
        f_name_tfidf = "map_file_bow_%d_tfidf.pkl" % len(self.solved_files)
        print "\n\nComputing query doc BoW vectors"
        if os.path.isfile(f_name) and self.cache_bool:
            with open(f_name, "r") as f:
                map_file_bow = pickle.load(f)
        if os.path.isfile(f_name_tfidf) and self.cache_bool:
            with open(f_name_tfidf, "r") as f:
                map_file_tfidf_bow = pickle.load(f)
        if not map_file_bow:
            for i, text in enumerate(self.solved_files_question_text):
                file_bow = bow_vector(text.split(' '), self.all_words, binary=False)
                map_file_bow[self.solved_files[i]] = file_bow
            if self.cache_bool:
                with open(f_name, "w") as f:
                    pickle.dump(map_file_bow, f)
        if not map_file_tfidf_bow:
            for i, text in enumerate(self.solved_files_question_text):
                file_bow = tfidf_bow_vector(text.split(' '), self.all_words, self.all_words_files)
                map_file_tfidf_bow[self.solved_files[i]] = file_bow
            if self.cache_bool:
                with open(f_name_tfidf, "w") as f:
                    pickle.dump(map_file_tfidf_bow, f)
        return map_file_bow

    def _get_map_file_bigram(self):
        bigram_vectorizer = CountVectorizer(ngram_range=(1,2), token_pattern=r'\b\w+\b', min_df=2, binary=False)
        map_file_bigram = dict()
        f_name = "map_file_bigram_%d_nonbinary.pkl" % len(self.solved_files)
        print "\n\nComputing query doc BiGram vectors"
        if os.path.isfile(f_name) and self.cache_bool:
            with open(f_name, "r") as f:
                map_file_bigram = pickle.load(f)
        if not map_file_bigram:
            # text is self.solved_files_question_text[i]
            # id is self.solved_files[i]
            qtext = self.solved_files_question_text
            bigram_vec = bigram_vectorizer.fit_transform(qtext)

            for i, qid in enumerate(self.solved_files):
                file_bigram = bigram_vec[i]
                map_file_bigram[qid] = file_bigram.toarray()[0]
            if self.cache_bool:
                with open(f_name, "w") as f:
                    pickle.dump(map_file_bigram, f)
        return map_file_bigram

    def _get_support_bigram(self):
        bigram_vectorizer = CountVectorizer(ngram_range=(1,2), token_pattern=r'\b\w+\b', min_df=2, binary=True)
        map_support_docs_bigram = dict()
        f_name = "map_support_docs_bigram_%d_%sbinary.pkl"%(len(self.solved_files), "" if self.use_binary_bow else "non")
        print "\n\nComputing support doc BiGram vectors"
        if os.path.isfile(f_name) and self.cache_bool:
            with open(f_name, "r") as f:
                map_support_docs_bigram = pickle.load(f)
        if not map_support_docs_bigram:
            support_text = list()
            for doc, super_title in self.map_support_docs_super_titles.iteritems():
                content_text = self.map_support_docs_text[doc]
                total_text = super_title+" "+content_text
                support_text.append(total_text)
            
            bigram_vec = bigram_vectorizer.fit_transform(support_text)

            for doc, super_title in self.map_support_docs_super_titles.iteritems():
                file_bigram = bigram_vec[i]
                map_support_docs_bigram[doc] = file_bigram.toarray()[0]
            if self.cache_bool:
                with open(f_name, "w") as f:
                    pickle.dump(map_support_docs_bigram, f)
        return map_support_docs_bigram

    def _get_support_bow(self):
        self.use_binary_bow = True
        map_support_docs_bow = {}
        f_name = "map_support_docs_bow_%d_%sbinary.pkl"%(len(self.solved_files), "" if self.use_binary_bow else "non")
        print "\n\nComputing support doc BoW vectors"
        if os.path.isfile(f_name) and self.cache_bool:
            with open(f_name, "r") as f:
                map_support_docs_bow = pickle.load(f)
        if not map_support_docs_bow:
            for doc, super_title in self.map_support_docs_super_titles.items():
                content_text = self.map_support_docs_text[doc]
                total_text = super_title+" "+content_text
                map_support_docs_bow[doc] = bow_vector(total_text.split(' '), self.all_words, binary=self.use_binary_bow)
            if self.cache_bool:
                with open(f_name, "w") as f:
                    pickle.dump(map_support_docs_bow, f)
        return map_support_docs_bow

    def _get_X_and_y_word_vectors(self):
        wv = WordVector()
        X = []
        y = []
        map_question_word_vectors = {}
        f_name = "map_question_word_vectors_%d_%d.pkl"%(len(self.solved_files),no_words)
        if os.path.isfile(f_name):
            with open(f_name,"r") as f:
                map_question_word_vectors = pickle.load(f)
                assert len(map_question_word_vectors) == len(self.solved_files)
        if not map_question_word_vectors:
            for i, text in enumerate(self.solved_files_question_text):
                text_list = text.split(' ')
                x_i = []
                x_i_add = [0.0 for _ in range(word_vector_dim)]
                for j in text_list:
                    if j not in self.all_words:
                        continue
                    x_i_j = wv.word_vector(j)
                    for k in x_i_j:
                        x_i.append(k)
                    for k in range(len(x_i_j)):
                        x_i_add[k] += x_i_j[k] 
                if not self.add_word_vecs: 
                    while len(x_i)<no_words*word_vector_dim:
                        x_i.append(0.0)
                    if len(x_i)>no_words*word_vector_dim:
                        x_i=x_i[:no_words*word_vector_dim]
                    assert len(x_i) == no_words*word_vector_dim
                assert len(x_i_add) == word_vector_dim

                if self.add_word_vecs:
                    x_i = x_i_add
                map_question_word_vectors[self.solved_files[i]] = x_i
                del x_i,x_i_add

                #cur_doc = self.solved_files_support_docs[self.solved_files[i]]
                #y.append(self.all_support_docs.index(cur_doc))
            #X = np.array(X)
            #y = np.array(y)
            print "Beginning to write down a map of size %d" %(len(map_question_word_vectors))
            #with open(f_name, "w") as f:
            #    pickle.dump(map_question_word_vectors, f)
        
        if map_question_word_vectors:
            for i, text in enumerate(self.solved_files_question_text):
                x_i = map_question_word_vectors[self.solved_files[i]]
                X.append(np.array(x_i))
                del x_i
                #del map_question_word_vectors[self.solved_files[i]]
                cur_doc = self.solved_files_support_docs[self.solved_files[i]]
                y.append(self.all_support_docs.index(cur_doc))
             
        X = np.array(X)
        y = np.array(y)
        with open(f_name, "w") as f:
            pickle.dump(map_question_word_vectors, f)
         
        return X, y

    def _get_X_and_y(self, map_file_bow):
        data = []
        row = []
        col = []
        for i, text in enumerate(self.solved_files_question_text):
            bow = map_file_bow[self.solved_files[i]]
            for j, val in enumerate(bow):
                if val != 0:
                    data.append(val)
                    row.append(i)
                    col.append(j)
        X = coo_matrix((data, (row, col)), shape=(len(self.solved_files_question_text), len(bow)), dtype=int).tocsr()
        y = []
        for solved_file in self.solved_files:
            cur_doc = self.solved_files_support_docs[solved_file]
            y.append(self.all_support_docs.index(cur_doc))
        y = np.array(y)
        print X.shape
        return X, y


    def _classifier_score(self, X, y, clf_name):
        """ 
        Return score over 5-fold cross validation for specified classifier


        X: numpy array of bag of words (or other features)
        y: numpy array of labels (supports multi-class)
        """
        # if using a neural network, we must change y to one-hot encoding
        if clf_name in ('CNN', 'NeuralNet'):
            y_vector = y
            y_one_hot = np.zeros((X.shape[0], len(self.all_support_docs)))
            for i, v in enumerate(y):
                y_one_hot[i, v] = 1
            if not self.run_without_word_vectors:
                y = y_one_hot
            else:
                y = coo_matrix(y_one_hot, dtype=bool).tocsr()

        rand_state = np.random.RandomState(seed=100)
        num_folds = 5
        results = []
        # for kfold_index, (train_index, test_index) in enumerate(KFold(n_splits=num_folds, shuffle=True, random_state=rand_state).split(X)):
        if True:
            kfold_index = 0
            train_index = np.arange(8000)
            test_index = np.arange(8000, 9553)
            # initialize model
            if clf_name == 'CNN':
                clf = Sequential()
                if self.run_without_word_vectors:
                    raise Exception("Can't do CNN without word vectors.")
                clf.add(Convolution1D(input_shape=(no_words*word_vector_dim,1),nb_filter=200,filter_length=5,border_mode='same',activation='sigmoid'))
                clf.add(Convolution1D(nb_filter=100,filter_length=3,border_mode='same',activation='sigmoid'))
                clf.add(GlobalMaxPooling1D())
                clf.add(Dense(input_dim=100,output_dim=len(self.all_support_docs),activation='softmax'))
                #sgd = SGD()
                #clf.compile(loss='categorical_crossentropy',optimizer=sgd)
                #adagrad = Adagrad()
                #clf.compile(loss='categorical_crossentropy',optimizer=adagrad)
                adam = Adam()
                clf.compile(loss='categorical_crossentropy',optimizer=adam)

            if clf_name == 'NeuralNet':
                clf = Sequential()
                if not self.run_without_word_vectors:
                    input_dim = word_vector_dim
                    if not self.add_word_vecs:
                        input_dim=word_vector_dim*no_words
                    clf.add(Dense(input_dim=input_dim,output_dim=word_vector_dim,activation='tanh'))
                    clf.add(Dense(input_dim=word_vector_dim,output_dim=len(self.all_support_docs),activation='softmax'))
                else:
                    #clf.add(Dense(input_dim=len(self.all_words),output_dim=len(self.all_words)='sigmoid'))
                    input_dim = X.shape[1]
                    output_dim = y.shape[1]
                    if self.bigram_bool:
                        clf.add(Dense(input_dim=input_dim,output_dim=output_dim,activation='softmax'))
                    else:
                        clf.add(Dense(input_dim=len(self.all_words),output_dim=len(self.all_support_docs),activation='softmax'))

                sgd = SGD()
                clf.compile(loss='categorical_crossentropy',optimizer=sgd)
                #adagrad = Adagrad()
                #clf.compile(loss='categorical_crossentropy',optimizer=adagrad)
                #adam = Adam()
                #clf.compile(loss='categorical_crossentropy',optimizer=adam)

            if clf_name == 'DecisionTree':
                clf = dtc(max_depth=30, max_features=None, min_samples_split=5)

            if clf_name == 'LogisticRegression':
                clf = lr()

            if clf_name == 'SVM':
                clf = SVC()

            # train and test model
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]

            if clf_name == 'CNN':
                X_train = X_train.reshape(X_train.shape+(1,))
                X_test = X_test.reshape(X_test.shape+(1,))
                clf.fit(X_train, y_train, nb_epoch=NUM_EPOCHS, batch_size=Batch_Size, verbose=2)
                if self.scoring_n == 1:
                    y_pred = clf.predict_classes(X_test, verbose=0)
                    correct = 0
                    for i, pred in enumerate(y_pred):
                        if y_test[i, pred] == 1:
                            correct += 1
                    acc = correct*1.0/len(y_pred)
                    print "Test accuracy for k-fold iteration %d of %d: %f"%(kfold_index, num_folds, acc)
                    results.append(acc)
                elif self.scoring_n > 1:
                    probs = clf.predict_proba(X_test, verbose=0)
                    best_n = np.argsort(probs, axis=1)[:,-self.scoring_n:]
                    correct = 0.
                    total = 0.
                    y_test_vec = y_vector[test_index]
                    for actual_val, top_guesses in zip(y_test_vec, best_n):
                        for guess in top_guesses:
                            if actual_val == guess:
                                correct += 1
                        total += 1
                    acc = correct / total
                    print "Test accuracy for k-fold iteration %d of %d (top %d): %f"%(kfold_index, num_folds, self.scoring_n, acc)
                    results.append(acc)
                else:
                    raise ValueError('Top K prediction value must be greater than 0: %d < 0' % self.scoring_n)
            elif clf_name == 'NeuralNet':
                if not self.run_without_word_vectors:
                    
                    clf.fit(X_train, y_train, nb_epoch=NUM_EPOCHS, batch_size=Batch_Size, verbose=2)
                    # if self.scoring_n == 1:
                    y_pred = clf.predict_classes(X_test, verbose=0)
                    # else:
                    y_probs = clf.predict_proba(X_test, verbose=0)
                else:
                    # for some reason, keras's fit function crashes with sparse matrix input...so convert back to array
                    clf.fit(X_train.toarray(), y_train.toarray(), nb_epoch=NUM_EPOCHS, batch_size=Batch_Size, verbose=2)
                    # if self.scoring_n == 1:
                    y_pred = clf.predict_classes(X_test.toarray(), verbose=0)
                    # else:
                    y_probs = clf.predict_proba(X_test.toarray(), verbose=0)

                if self.scoring_n == 1:
                    correct = 0
                    for i, pred in enumerate(y_pred):
                        if y_test[i, pred] == 1:
                            correct += 1
                    acc = correct*1.0/len(y_pred)
                    print "Test accuracy for k-fold iteration %d of %d: %f"%(kfold_index, num_folds, acc)
                    results.append(acc)
                elif self.scoring_n > 1:
                    total = 0.
                    correct = 0.
                    mrr_float = 0.
                    all_best = np.argsort(y_probs, axis=1)[:,-len(self.all_support_docs):]
                    best_n = np.argsort(y_probs, axis=1)[:,-self.scoring_n:]
                    y_test_vec = y_vector[test_index]

                    error_analysis_file = open("Error_analysis_file.txt","w")
                    ranks_file = open("ranks_file.txt","w")
                    for actual_val, top_guesses in zip(y_test_vec, best_n):
                        found = False
                        for guess in top_guesses:
                            if actual_val == guess:
                                correct += 1
                                found = True
                                break
                        if not found:
                            support_doc_name = self.all_support_docs[actual_val]
                            error_analysis_file.write(support_doc_name+"\t"+self.map_support_docs_super_titles[support_doc_name]+'\n')
                        total += 1
                    acc = correct / total
                    error_analysis_file.close()
                    print "Test accuracy for k-fold iteration %d of %d (top %d): %f"%(kfold_index, num_folds, self.scoring_n, acc)
                    results.append(acc)
                    for actual_val,top_guesses in zip(y_test_vec, all_best):
                        if actual_val not in top_guesses:
                            mrr_float += 0.
                        else:
                            mrr_float += 1.0/(len(self.all_support_docs) - top_guesses.tolist().index(actual_val))
                            ranks_file.write(str(len(self.all_support_docs) - top_guesses.tolist().index(actual_val))+'\n')
                            print mrr_float
                    print "MRR value is %f " %(mrr_float/len(y_probs))
                    ranks_file.close()
                else:
                    raise ValueError('Top K prediction value must be greater than 0: %d < 0' % self.scoring_n)

            elif clf_name == 'Majority':
                test_N = len(y_test)
                majority_guess = most_common(y_train) * np.ones(test_N)

                acc = sum(y_test == majority_guess) / float(test_N)
                print "Test accuracy for k-fold iteration %d of %d: %f"%(kfold_index, num_folds, acc)
                
                results.append(acc)

            else:
                clf.fit(X_train, y_train)
                if self.scoring_n == 1:
                    acc = clf.score(X_test, y_test)
                    print "Test accuracy for k-fold iteration %d of %d: %f"%(kfold_index, num_folds, acc)
                    results.append(acc)
                elif self.scoring_n > 1:
                    # check if answers in y_test were in top K of predict_probs
                    probs = clf.predict_proba(X_test)
                    best_n = np.argsort(probs, axis=1)[:,-self.scoring_n:]
                    correct = 0.
                    total = 0.
                    out = open("tao_test_logreg_purged.txt", "w")
                    for t_ind, (actual_val, top_guesses) in enumerate(zip(y_test, best_n)):
                        for guess in top_guesses:
                            if actual_val == guess:
                                correct += 1
                                question = self.solved_files[len(train_index)+t_ind][self.solved_files[len(train_index)+t_ind].rfind("/")+1:]
                                cor_doc = self.all_support_docs[actual_val]
                                candidates = np.array(self.all_support_docs)[top_guesses]
                                out.write("%s\t%s\t%s\n"%(question, cor_doc, (" ".join(candidates)).strip()))
                                break
                        total += 1
                    out.close()
                    acc = correct / total
                    print "Test accuracy for k-fold iteration %d of %d (top %d): %f"%(kfold_index, num_folds, self.scoring_n, acc)
                    results.append(acc)
                else:
                    raise ValueError('Top K prediction value must be greater than 0: %d < 0' % self.scoring_n)

            del clf
        return np.mean(results)

    def _classifier_error_analysis(self, X, y):
        clf = lr() # use our best classifier baseline for error analysis: LogisticRegression
        k = int(len(X)*0.8)
        X_train = X[:k]
        y_train = y[:k]
        X_test = X[k:]
        y_test = y[k:]
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        print "Accuracy: %f"%(sum(y_pred == y_test) * 1.0 / len(y_pred))
        # compute most common docs
        print
        print "Top 10 most common predicted docs:"
        for val, count in collections.Counter(y_pred).most_common(10):
            print "%s has count %d"%("https://support.apple.com/en-us/" + self.all_support_docs[val], count)
        print
        print "Top 10 most common actual docs:"
        for val, count in collections.Counter(y_test).most_common(10):
            print "%s has count %d"%("https://support.apple.com/en-us/" + self.all_support_docs[val], count)

        # compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.matshow(cm/np.linalg.norm(cm))
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted doc index")
        plt.ylabel("Actual doc index")
        plt.show()

        print
        # case-by-case error analysis
        for i in range(len(y_pred)):
            if i % 10 == 0:
                print "For i = %d, opened 3 tabs: query, then actual support doc, then predicted support doc"%i
                webbrowser.open_new_tab("https://discussions.apple.com/thread/" + self.solved_files[i+k].split("/")[-1])
                webbrowser.open_new_tab("https://support.apple.com/en-us/" + self.all_support_docs[y_test[i]])
                webbrowser.open_new_tab("https://support.apple.com/en-us/" + self.all_support_docs[y_pred[i]])
                raw_input("Press <enter> to continue")
        print "Done. Quitting..."
        sys.exit(1)

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def main():
    usage = "python classifier.py [-n <data size>] [-e to do error analysis] [-w <run without word vectors>]  [-t to score based on correct doc in top T predictions] [-b <use bigrams too>] [-l <lemmatization>] [-c <do NOT use caching>] [-a <add word vectors>]"
    data_size = sys.maxint
    ea = False
    run_without_word_vectors = False
    top_n = 1
    binary_bool = False
    lemma_bool = False
    cache_bool = True
    add_word_vecs = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:aewt:blc")
    except getopt.GetoptError:
        print usage
        sys.exit(1)
    for opt, arg in opts:
        if opt == "-n":
            data_size = int(arg)
        if opt == "-e":
            ea = True
        if opt == "-w":
            run_without_word_vectors = True
        if opt == '-t':
            top_n = int(arg)
        if opt == '-b':
            binary_bool = True
            run_without_word_vectors = False
        if opt == '-l':
            lemma_bool = True
            cache_bool = False
        if opt == '-c':
            cache_bool = False
        if opt == '-a':
            add_word_vecs = True

    if data_size < sys.maxint:
        print "Only using %d samples."%data_size
    else:
        print "Using full dataset."
    if ea:
        print "Running error analysis mode."
    if binary_bool:
        print 'Using bigrams AND unigrams'
    print
    b = ClassifierBaseline(data_size=data_size, do_error_analysis=ea, run_without_word_vectors = run_without_word_vectors, scoring_n=top_n, bigram_bool=binary_bool, lemma_bool=lemma_bool, cache_bool=cache_bool, add_word_vecs = add_word_vecs)
    b.populate_data()
    b.run_classifiers()

if __name__ == "__main__":
    main()
