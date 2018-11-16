from baseline import *
import re
import lucene
import tensorflow as tf
import random
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.index import IndexReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version
from nltk.tokenize import sent_tokenize, word_tokenize
from util import WordVector

WORD_VEC_DIM = 200

class LuceneRankerBaseline(Baseline):

    def set_lucene(self):
        lucene.initVM()
        analyzer = StandardAnalyzer(Version.LUCENE_4_9)
        reader = IndexReader.open(SimpleFSDirectory(File("lucene/index/")))
        self.searcher = IndexSearcher(reader)
        self.qp = QueryParser(Version.LUCENE_4_9, "doc_text", analyzer)

    def do_lucene_ranker_baseline(self, for_test=True):
        correct = 0
        sentence_count = 0
        print
        considered = 0
        important_words = set(self.all_support_docs_words)
        dissimilar_word_map = get_dissimilar_words(self)
        top_n = 50
        file_name_write = "tao_test_lucene_"+str(top_n)+".txt"
        if not for_test:
            file_name_write = "tao_train_lucene_"+str(top_n)+".txt"
        out = open(file_name_write, "w")
        file_analyse = open('analysis.txt','w')
        f_ranks = open('ranks.txt','w')
        ranks = {}
        for j, name in enumerate(self.solved_files):
            if name in self.solved_files_with_support_docs:
                if not for_test:
                    continue
            if name not in self.solved_files_with_support_docs:
                if for_test:
                    continue
            considered += 1
            i = j
            s = self.solved_files_question_text[i]
            #spaces = [m.start() for m in re.finditer(" ", s)]
            max_words = 100
            #if len(spaces) > max_words:
            #    s = s[:spaces[max_words]]
            s_list = s.split()
            s_list_new = []
            for s_i in s_list:
                if s_i in important_words:
                    s_list_new.append(s_i)
                if s_i in dissimilar_word_map and dissimilar_word_map[s_i]:
                    s_list_new.append(dissimilar_word_map[s_i])
            s=' '.join(s_list_new[:max_words]).strip()
            #s = self.solved_files_question_content[i]
            s_list = sent_tokenize(s)
            sentence_count += len(s_list)
            s = ' '.join([s_small for s_small in s.split() if s_small.isalnum()][:100]).strip()
            if not s:
                #continue
                top_pred_list = []
            else:
                top_pred_list = self.get_top_for_query(s,top_n)
            actual = self.solved_files_support_docs[name]
            if for_test:
                if actual in top_pred_list:
                    #file_analyse.write(pred+'\n'+self.map_support_docs_titles[pred]+'\n')
                    correct += 1
                    question = self.solved_files[i][self.solved_files[i].rfind("/")+1:]
                    out.write("%s\t%s\t%s\n"%(question, actual, (" ".join(top_pred_list)).strip()))
                    cur_rank = top_pred_list.index(actual)
                    if cur_rank not in ranks:
                        ranks[cur_rank] = 1
                    else:
                        ranks[cur_rank] = ranks[cur_rank] + 1
                    f_ranks.write(name[name.rfind('/')+1:]+','+str(cur_rank+1)+'\n')
                else:
                    cur_rank = len(self.all_support_docs)
                    if cur_rank not in ranks:
                        ranks[cur_rank] = 1
                    else:
                        ranks[cur_rank] = ranks[cur_rank] + 1
                    f_ranks.write(name[name.rfind('/')+1:]+','+str(len(self.all_support_docs))+'\n')

            else:
                if actual not in top_pred_list:
                    top_pred_list.append(actual)
                else:
                    correct += 1
                question = self.solved_files[i][self.solved_files[i].rfind("/")+1:]
                out.write("%s\t%s\t%s\n"%(question, actual, (" ".join(top_pred_list)).strip()))

            if (j+1) % 100 == 0:
                print "So far %d correct out of %d"%(correct, j+1)
        out.close()
        acc = correct*1.0/considered
        print "\nFinal ccuracy: %f"%acc
        file_analyse.close()
        print "%f Number of sentences per text" %(sentence_count*1.0/(considered))
        print ranks
        #print (np.sum(np.array(ranks.values()))+np.sum(np.array(ranks.keys())))/len(ranks.keys())
        avg_ranks = 0.
        for r,c in ranks.items():
            avg_ranks += (1+r)*c
        avg_ranks /= considered*1.0
        print "avg ranks %f" %avg_ranks
        f_ranks.close()
        return acc

    def get_top_for_query(self,query,top_n):
        
        query = self.qp.parse(QueryParser.escape(query))
        hits = self.searcher.search(query, top_n)
        if not hits:
            return []
        top_pred_list = []
        for j in range(min(top_n,len(hits.scoreDocs))):
            hit = hits.scoreDocs[j]
            pred = self.searcher.doc(hit.doc).get("doc_name")
            top_pred_list.append(pred)
        return top_pred_list

    def get_subset_query_word(self, file_name, top_n, choose):
        file_index = self.solved_files.index(file_name)
        question_content = self.solved_files_question_content[file_index]
        question_words = word_tokenize(question_content)
        modified_question = []
        for i,word in enumerate(question_words):
            if choose[i] == 1:
                modified_question.append(word)
        if not modified_question:
            return []
        modified_query = ' '.join(modified_question[:1000]).strip()
        return self.get_top_for_query(modified_query, top_n)


    def get_subset_query(self, file_name, top_n, choose):
        
        file_index = self.solved_files.index(file_name)
        question_content = self.solved_files_question_content[file_index]
        question_sentences = sent_tokenize(question_content)
        modified_question = []
        for i,sent in enumerate(question_sentences):
            if choose[i] == 1:
                for word in sent.split():
                    modified_question.append(word)
        if not modified_question:
            return []
        #    modified_question = self.solved_files_question_label[file_index].split()
        modified_query = ' '.join(modified_question[:1000]).strip()
        return self.get_top_for_query(modified_query, top_n)

    def get_train_data(self):
        
        reward_all = 0.0
        reward_random = 0.0
        for i,solved_file in enumerate(self.solved_files):
            
            question_content = self.solved_files_question_content[i]
            question_list = sent_tokenize(question_content)
            choose = []
            for j in range(len(question_list)):
                random_float = random.random()
                if random_float < 0.5:
                    choose.append(0)
                else:
                    choose.append(1)
            top_pred_list = self.get_subset_query(solved_file,10,choose)
            correct_doc = self.solved_files_support_docs[solved_file]
            if correct_doc in top_pred_list:
                reward_random += 1.0/(1+top_pred_list.index(correct_doc))
            choose_identity = [1 for _ in range(len(question_list))]
            top_pred_list = self.get_subset_query(solved_file,10,choose_identity)
            correct_doc = self.solved_files_support_docs[solved_file]
            if correct_doc in top_pred_list:
                reward_all += 1.0/(1+top_pred_list.index(correct_doc))
        
        print reward_all,reward_random

    
    def weight_decay_penalty(self, weights, penalty):
        return penalty * sum([tf.nn.l2_loss(w) for w in weights])

    def get_choices(self, sess, solved_files_question_content):
        #choices = []
        #for I,solved_file_text in enumerate(solved_files_question_content_list):
        i = self.solved_files_question_content_list.index(solved_files_question_content)
        ques_list = sent_tokenize(solved_files_question_content)
        choose = []
        for j in range(len(ques_list)):
            ques_list_bow = bow_vector(ques_list[j].split(' '), self.all_words, binary=False)
            ques_list_bow = np.array(ques_list_bow).reshape(1, len(self.all_words))
            temp_out = sess.run([Qabs], feed_dict={inputs1:ques_list_bow})
            temp_out = temp_out[0][0][0]
            if temp_out < 0.5:
                choose.append(0)
            else:
                choose.append(1)
        return choose

    
    def set_tensorflow(self):
        
        tf.reset_default_graph()
        inputs1 = tf.placeholder(shape=[1, len(self.all_words)], dtype=tf.float32)
        W = tf.Variable(tf.random_uniform([len(self.all_words), 100],0,0.01))
        Hmid = tf.nn.relu(tf.matmul(inputs1,W))
        Wout = tf.Variable(tf.random_uniform([100,2],0,0.01))
        Qabs = tf.nn.softmax(tf.matmul(Hmid,Wout))

        Q = tf.reshape(Qabs,[2])
        index = tf.placeholder(shape=[1], dtype=tf.int32)
        reward = tf.placeholder(shape=[1], dtype=tf.float32)
        responsible_weight = tf.slice(Q,index,[1])
        error = tf.log(responsible_weight)*reward
        l2_norm = self.weight_decay_penalty([W,Wout],1.0)

        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        updateModel = trainer.minimize(error)

        init = tf.initialize_all_variables()
        top_n = 50

        wv = WordVector()

        with tf.Session() as sess:
            sess.run(init)
            for k in range(500):
                print "Beginning epoch num %d" %(k+1)
                train_correct = 0
                train_mrr = 0.
                for i,solved_file_text in enumerate(self.solved_files_question_content[:8000]):
                    if (i+1)%100 == 0:
                        print "File being trained %d" %(i+1)
                    ques_list = sent_tokenize(solved_file_text)
                    oldW = []
                    oldWout = []
                    choose = []
                    for j in range(len(ques_list)):
                        ques_list_bow = bow_vector(ques_list[j].split(' '), self.all_words, binary=False)
                        assert len(ques_list_bow) == len(self.all_words)
                        all_zeroes = True
                        for l in ques_list_bow:
                            if l != 0.0:
                                all_zeroes = False
                                break
                        if all_zeroes:
                            print "Sentence %d from question %d has all zeros" %(j+1,i+1)
                        ques_list_bow = np.array(ques_list_bow).reshape(1, len(self.all_words))
                        temp_out = sess.run([Qabs], feed_dict={inputs1:ques_list_bow})
                        print temp_out
                        temp_out = temp_out[0][0]
                        print temp_out
                        random_float = random.random()
                        print "Considering word %s" %ques_list[j].split(' ')
                        if temp_out[0] < temp_out[1]:
                            print "Did not pick it"
                            choose.append(0)
                        else:
                            print "Picked it"
                            choose.append(1)
                    solved_file = self.solved_files[i]
                    top_pred_list = self.get_subset_query(solved_file,top_n,choose)
                    top_pred_list_ideal = self.get_subset_query(solved_file,top_n,[1 for _ in range(len(ques_list))])
                    cor_doc = self.solved_files_support_docs[solved_file]
                    r = 0.0
                    if cor_doc in top_pred_list:
                        print "Gets top 50 result"
                        r = -1.0/(1.0 + top_pred_list.index(cor_doc)) 
                        train_correct += 1
                        train_mrr -= r
                        if cor_doc in top_pred_list_ideal:
                            r += 1.0/(1.0 + top_pred_list_ideal.index(cor_doc))
                    else:
                        print "Misses top 50 result"
                        r = 1.0
                    if cor_doc in top_pred_list_ideal:
                        r += 1.0/(1.0 + top_pred_list_ideal.index(cor_doc))
                    r = r/len(ques_list)
                    if r <0:
                        print "Positive reward %f for sentence number %d" %(r,j+1)
                        r = r*100
                    for j in range(len(choose)):
                        if choose[j] == 1 or choose[j] == 0:
                            ques_list_bow = bow_vector(ques_list[j].split(' '), self.all_words, binary=False)
                            ques_list_bow = np.array(ques_list_bow).reshape(1, len(self.all_words))
                            print "Considering word %s" %ques_list[j].split(' ')
                            ind = 1
                            if choose[j] == 1:
                                ind = 0
                            else:
                                ind = 1
                            _,W1,W2,norm = sess.run([updateModel,W,Wout,l2_norm], feed_dict={inputs1:ques_list_bow,reward:[r],index:[ind]})
                            if (len(oldW) > 0 and oldW.all() != W1.all()):
                                print "There has been a weight update in Weight1"
                            elif (len(oldWout) > 0 and oldWout.all() != W2.all()):
                                print "There has been a weight update in Weight2"
                            else:
                                print "No weight update"
                            oldW = W1
                            oldWout = W2
                            print "Norm of weights %f" %norm
                
                print "Train Correct %d" %train_correct
                print "Train mrr %f" %train_mrr


                ideal_correct = 0
                correct = 0
                default_correct = 0
                mrr_correct = 0.0
                mrr_default = 0.0
                mrr_ideal = 0.0
                max = 0
                min = 1
                print "Beginning Testing"
                for I,solved_file_text in enumerate(self.solved_files_question_content[8000:]):
                    i = I+8000
                    #i = I
                    ques_list = sent_tokenize(solved_file_text)
                    choose = []
                    for j in range(len(ques_list)):
                        ques_list_bow = bow_vector(ques_list[j].split(' '), self.all_words, binary=False)
                        all_zeroes = True
                        for l in ques_list_bow:
                            if l != 0.0:
                                all_zeroes = False
                                break
                        if all_zeroes:
                            print "Sentence %d from question %d has all zeros" %(j+1,i+1)                                                                                                                                                                                                                               
                        ques_list_bow = np.array(ques_list_bow).reshape(1, len(self.all_words))
                        output = sess.run([Qabs], feed_dict={inputs1:ques_list_bow})
                        print output[0][0]
                        if output[0][0][0] < min:
                            min = output[0][0][0]
                        if output[0][0][0] > max:
                            max = output[0][0][0]
                        if output[0][0][0]<output[0][0][1]:
                            print "File num %d getting line num %d kicked out" %(i+1,j+1)
                            choose.append(0)
                        else:
                            print "File num %d keeping line num %d as is" %(i+1,j+1)
                            choose.append(1)
                    solved_file = self.solved_files[i]        
                    top_pred_list = self.get_subset_query(solved_file,top_n,choose)
                    cor_doc = self.solved_files_support_docs[solved_file]
                    if cor_doc in top_pred_list:
                        correct += 1
                        mrr_correct += 1.0/(1+top_pred_list.index(cor_doc))
                    top_pred_def_list = self.get_subset_query(solved_file,top_n,[1 for _ in range(len(ques_list))])
                    if cor_doc in top_pred_def_list:
                        default_correct += 1
                        mrr_default += 1.0/(1+top_pred_def_list.index(cor_doc))
                    solved_file_content = self.solved_files_question_content[i]
                    solved_file_words = solved_file_content.split()
                    solved_file_shorter_words = []
                    for word in solved_file_words:
                        if word in self.all_support_docs_words:
                            solved_file_shorter_words.append(word)
                    top_pred_ideal_list = self.get_top_for_query(' '.join(solved_file_shorter_words).strip(),top_n)
                    if cor_doc in top_pred_ideal_list:
                        ideal_correct += 1
                        mrr_ideal += 1.0/(1+top_pred_ideal_list.index(cor_doc))
                    
                print "Maximum %f and minimum %f" %(max,min)
                print "Test Correct %d, default_correct %d and ideal_correct %d" %(correct,default_correct,ideal_correct)
                print "Test Correct mrr %f, default_mrr %f and ideal_mrr %f" %(mrr_correct,mrr_default,mrr_ideal)


def lucene_test():
    usage = "python lucene_ranker.py [-n <data size>] [-d to download missing support documents]"
    data_size = sys.maxint
    download_support_docs = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:d")
    except getopt.GetoptError:
        print usage
        sys.exit(1)
    for opt, arg in opts:
        if opt == "-n":
            data_size = int(arg)
        elif opt == "-d":
            download_support_docs=True
    if data_size < sys.maxint:
        print "Only using %d samples."%data_size
    else:
        print "Using full dataset."
    if download_support_docs:
        print "Will crawl missing support documents."
    else:
        print "Will not crawl missing support documents."
    print
    b = LuceneRankerBaseline(data_size=data_size, download_support_docs=download_support_docs)
    b.populate_data()
    b.set_lucene()
    b.do_lucene_ranker_baseline()

if __name__ == "__main__":
    #lucene_test()
    #get_train_data()
    data_size = sys.maxint
    download_support_docs = False
    b = LuceneRankerBaseline(data_size=data_size, download_support_docs=download_support_docs)
    b.populate_data()
    b.set_lucene()
    #b.set_tensorflow()
    b.do_lucene_ranker_baseline(for_test=True)
