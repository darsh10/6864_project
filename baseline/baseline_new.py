
import random
import re
import sys
sys.path.append('..')
import getopt
from time import sleep
import shlex, subprocess
from os import listdir
import os
from os.path import isfile, join
import string
import pickle
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import wordpunct_tokenize, word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob as tb
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from sklearn.cluster import KMeans

from parsing.apple_discussion_threads import get_chat_data
from parsing.apple_support_docs import getTitle
from parsing.apple_support_docs import getText,getDocName,getExactTitle,getExactSubTitle
from parsing.apple_support_docs import getSoup,getCompleteTextWithoutTitle,getParagraphTitlesBodies
from parsing.apple_support_docs import getSmallTitles,getTitleSubtitle,getTextWithoutTitle,getTextWithoutTitleSentences
from util import *
import gensim
from gensim import corpora, models
from parsing.apple_user_questions import get_question_answer,getSoup

#paths to discussion threads (DATA) and to support documents (docs)
DATA_PATH = "../DATA_short/"
DOC_PATH = "../docs_short/"

SUPPORT_DOC_MIN = 10


class Baseline(object):
    def __init__(self, data_size, do_error_analysis=False, download_support_docs=False, run_without_word_vectors = False, scoring_n=1, bigram_bool=False, lemma_bool=False, cache_bool=True, add_word_vecs=False):
        # list(str) of discussion threads in the DATA folder
        self.discussion_files = sorted([join(DATA_PATH,f) for f in listdir(DATA_PATH) if isfile(join(DATA_PATH,f))])[:data_size]
        # list(str) of all support documents in the docs folder 
        self.support_doc_files = [f for f in listdir(DOC_PATH) if isfile(join(DOC_PATH,f))]
        self.run_without_word_vectors = run_without_word_vectors
        self.scoring_n = scoring_n
        self.bigram_bool = bigram_bool
        self.lemma_bool = lemma_bool
        self.cache_bool = cache_bool
        self.add_word_vecs = add_word_vecs
        # relevant_files is a list(str) of discussion thread names which are the only ones which have a support document as a reply and are correctly answered sometime
        try:
            self.relevant_files = eval(open("relevant_files.txt").read())[:data_size]
        except IOError:
            self.relevant_files = []
        print "Num discussion files found: %d" %(len(self.discussion_files))
        self.old_test_files = eval(open("new_relevant_files_solved_with_support_docs").read())
        self.new_test_files = []
        # num_solved is an int for the number of discussion threads that are finally solved
        self.num_solved = 0
        # num_discussions_with_support_docs is an int for the number of discussion threads with support docs in their answers and which are solved sometime
        self.num_discussions_with_support_docs = 0
        # num_discussions_with_support_docs_and_solved is an int for the number of discussion threads that are solved with a support doc in the solved answer itself
        self.num_discussions_with_support_docs_and_solved = 0
        self.solved_files_with_support_docs = []
        # list(str) of all names of support documents
        self.all_support_docs = []
        # list(str) of all support document links
        self.ignore_support_docs = set()
        self.all_support_doc_links = set()

        # list(str) of discussion threads names that were solved and that had support documents as a part of their solution sometime
        # the length of this list equals self.num_discussions_with_support_docs from above
        self.solved_files=[]
        # list(str) of question labels of above discussion threads
        self.solved_files_question_label=[]
        # list(str) of question body of above discussion threads
        self.solved_files_question_content=[]
        # list(str) of question label+" "+body of above discussion threads
        self.solved_files_question_text=[]
        # dictionary(str) to (str) of discussion thread name(str) to particular name(str) of support document that has been presented as a helper to the question
        self.solved_files_question_answer=[]
        self.solved_files_question_topic=[]
        self.solved_files_support_docs={}

        # num_support_docs_without_titles(int) name stands for itself
        self.num_support_docs_without_titles = 0
        # map(str) to (str) from support doc name to super_titles in support docs
        self.map_support_docs_super_titles = {}
        # map(str) to (str) from support doc name to lower-cased text (title and body) in support docs
        self.map_support_docs_text = {}
        # map(str) to (str) from support doc name to lower-cased title in support docs
        self.map_support_docs_titles = {}
        self.map_support_docs_title_subtitle = {}
        self.map_support_docs_text_list = {}
        self.map_support_docs_text_sentences_list = {}
        self.map_support_docs_complete_text = {}
        self.map_support_docs_complete_text_list = {}
        self.map_support_docs_paragraph_titles_bodies = {}
        self.map_support_docs_exact_sub_title = {}
        self.map_support_docs_exact_title = {}

        self.do_error_analysis = do_error_analysis
        self.download_support_docs = download_support_docs

        if self.lemma_bool:
            from nltk.stem.wordnet import WordNetLemmatizer
            self.lmtzr = WordNetLemmatizer()

    def _lemmatize(self, data):
        words = data.lower().split()
        return ' '.join([self.lmtzr.lemmatize(i) for i in words])

    def populate_data(self):
        redundant_map = pickle.load(open("redundant_map.pkl"))
        map_file_doc = pickle.load(open("../parsing/map_file_doc.p"))
        discussion_file_map = {}
        try:
            discussion_file_map = pickle.load(open("discussion_file.p","r"))
        except IOError:
            print "discussion file not found"
        print "Number of files in map_file_doc %d" %len(map_file_doc)
        for f in self.discussion_files:
            #if self.relevant_files and f not in self.relevant_files:
            #    continue
            #topic,question_author,question_label,question_content,answers,answer_authors,is_solved,\
            #    solved_answer,question_asked,question_solver,asked_question_string,asked_index = get_chat_data(f)
            question_label = question_content = question_answer_clean = question_answer = question_answer_link_referred = question_text_sent_list = question_topic = ""
            if f not in discussion_file_map:
                print "Will parse %s file's data" %f
                discussion_file_soup = getSoup(f)
                question_label,question_content,question_answer_clean,question_answer,question_answer_link_referred,question_text_sent_list,question_topic = get_question_answer(discussion_file_soup)
            else:
                question_label,question_content,question_answer_clean,question_answer,question_answer_link_referred,question_text_sent_list,question_topic = discussion_file_map[f]
            discussion_file_map[f] = tuple([question_label,question_content,question_answer_clean,question_answer,question_answer_link_referred,question_text_sent_list,question_topic])
            #if question_label.strip() == "":
            #    continue
            #answers_text = ' '.join(answers)
            if '../DATA/'+f[f.rfind('/')+1:] in self.old_test_files and f[f.rfind('/')+1:] in map_file_doc:
                self.new_test_files.append(f)
            if f[f.rfind('/')+1:] in map_file_doc:
                if "HT" not in map_file_doc[f[f.rfind('/')+1:]]:
                    continue
                self.solved_files.append(f)
                question_label = str(question_label)
                question_content = str(question_content)
                if self.lemma_bool:
                    qlabel = self._lemmatize(question_label)
                else:
                    qlabel = question_label.lower()
                self.solved_files_question_label.append(qlabel)

                if self.lemma_bool:
                    qcontent = self._lemmatize(question_content)
                else:
                    qcontent = question_content.lower()
                self.solved_files_question_content.append(qcontent)

                doc_name = map_file_doc[f[f.rfind('/')+1:]]
                question_text = question_label+" "+question_content

                if self.lemma_bool:
                    qtext = self._lemmatize(question_text)
                else:
                    qtext = question_text.lower()
                self.solved_files_question_text.append(qtext)
                self.solved_files_question_answer.append(question_answer_clean)
                self.solved_files_question_topic.append(question_topic)
                question_text_words = qtext.split(' ')
                self.solved_files_support_docs[f] = doc_name
                if doc_name not in self.all_support_docs:
                    self.all_support_docs.append(doc_name)
                self.num_discussions_with_support_docs+=1
            question_total = question_label+" "+question_content

            #if is_solved==1:
            #    self.num_solved+=1
        pickle.dump(discussion_file_map,open("discussion_file.p","w"))
        with open("new_relevant_files_solved_with_support_docs","w") as f:
            f.write(str(self.old_test_files))
        with open("new_relevant_files.txt", "w") as f:
            f.write(str(self.solved_files))
        print "Number of support docs: %d"%len(self.all_support_docs)
        print "Number of solved discussions: %d"%len(self.solved_files)
        assert len(self.solved_files) == self.num_discussions_with_support_docs
        print "Number of solved discussions with support articles: %d"%self.num_discussions_with_support_docs_and_solved
        print "Of %d old test files, %d new ones found" %(len(self.old_test_files),len(self.new_test_files))

        self._understand_support_doc_text()
        self._handle_super_titles()
        
        self.ignore_support_docs = list(self.ignore_support_docs)
        for ignore_doc in self.ignore_support_docs:
            if ignore_doc in self.all_support_docs:
                self.all_support_docs.remove(ignore_doc)
                if ignore_doc in self.map_support_docs_text.keys():
                    del self.map_support_docs_text[ignore_doc]
                if ignore_doc in self.map_support_docs_super_titles.keys():
                    del self.map_support_docs_super_titles[ignore_doc]
        self.ignore_support_docs = set(self.ignore_support_docs)        
        for file_name, support_doc in self.solved_files_support_docs.items():
            if support_doc not in self.ignore_support_docs:
                continue
            del self.solved_files_support_docs[file_name]
            ignore_ind = self.solved_files.index(file_name)
            del self.solved_files[ignore_ind]
            del self.solved_files_question_text[ignore_ind]
            del self.solved_files_question_label[ignore_ind]
            del self.solved_files_question_content[ignore_ind]
            del self.solved_files_question_answer[ignore_ind]
            del self.solved_files_question_topic[ignore_ind]

        #pickle.dump(discussion_file_map,open("discussion_file.p","w"))

        self.all_words,self.all_solved_question_words,self.all_support_docs_words,self.all_support_docs_title_words,self.all_words_map,self.all_words_files = self._create_all_words_set()
        assert len(self.all_words_map) == len(self.all_words)
        assert len(self.all_words_files) == len(self.all_words)

    def _handle_super_titles(self):
        try:
            print "Aaha Darshu, short cut for super titles"
            self.map_support_docs_titles = pickle.load(open("map_support_docs_titles.p","r"))
            self.map_support_docs_super_titles = pickle.load(open("map_support_docs_super_titles.p","r"))
            self.map_support_docs_title_subtitle = pickle.load(open("map_support_docs_title_subtitle.p","r"))
            self.map_support_docs_text_list = pickle.load(open("map_support_docs_text_list.p","r"))
            self.map_support_docs_text_sentences_list = pickle.load(open("map_support_docs_text_sentences_list.p","r"))
            self.map_support_docs_complete_text = pickle.load(open("map_support_docs_complete_text.p","r"))
            self.map_support_docs_complete_text_list = pickle.load(open("map_support_docs_complete_text_list.p","r"))
            self.map_support_docs_exact_title = pickle.load(open("map_support_docs_exact_title.p","r"))
            self.map_support_docs_exact_sub_title = pickle.load(open("map_support_docs_exact_sub_title.p","r"))
            self.map_support_docs_paragraph_titles_bodies = pickle.load(open("map_support_docs_paragraph_titles_bodies.p","r"))
            self.ignore_support_docs = pickle.load(open("handle_super_titles_ignore_support_docs","r"))
            return
        except IOError:
            print "Will create maps from handle super titles"
        for i in range(len(self.all_support_docs)):
            curr_doc = DOC_PATH+self.all_support_docs[i]
            soup=getSoup(curr_doc)
            if not soup:
                self.num_support_docs_without_titles+=1
                self.ignore_support_docs.add(self.all_support_docs[i])
                continue
            title = getTitle(soup)
            if not title:
                self.num_support_docs_without_titles+=1
                print self.all_support_docs[i], "has no title"
                self.ignore_support_docs.add(self.all_support_docs[i])
                continue
            title=title.lower()
            small_titles = getSmallTitles(soup)
            super_title=title.lstrip().rstrip()
            if small_titles:
                for small_title in small_titles:
                    s_t = ""
                    if small_title.findAll(text=True):
                        s_t = small_title.findAll(text=True)[0]
                        s_t = s_t.encode('ascii', 'ignore').decode('ascii')
                        s_t=s_t.split('.')[0].lower()
                        if "learn more" in s_t.lower():
                            break
                        if "additional product support information" in s_t.lower():
                            break
                        if "get help" in s_t.lower():
                            break
                    super_title=super_title+" "+s_t.lstrip().rstrip()
            title_subtitle = getTitleSubtitle(soup)
            self.map_support_docs_titles[self.all_support_docs[i]]=str(title)
            self.map_support_docs_super_titles[self.all_support_docs[i]]=str(super_title)
            self.map_support_docs_title_subtitle[self.all_support_docs[i]]=str(title_subtitle)
            self.map_support_docs_text_list[self.all_support_docs[i]] = getTextWithoutTitle(soup)
            self.map_support_docs_text_sentences_list[self.all_support_docs[i]] = getTextWithoutTitleSentences(soup)
            self.map_support_docs_complete_text[self.all_support_docs[i]] = getCompleteTextWithoutTitle(soup)
            self.map_support_docs_complete_text_list[self.all_support_docs[i]] = sent_tokenize(self.map_support_docs_complete_text[self.all_support_docs[i]])
            self.map_support_docs_exact_title[self.all_support_docs[i]] = str(getExactTitle(soup))
            self.map_support_docs_exact_sub_title[self.all_support_docs[i]] = str(getExactSubTitle(soup))
            if not getParagraphTitlesBodies(soup):
                print "No para in %s" %self.all_support_docs[i]
            self.map_support_docs_paragraph_titles_bodies[self.all_support_docs[i]] = tuple(getParagraphTitlesBodies(soup))
        print "A total of %d docs have no titles"%self.num_support_docs_without_titles
        pickle.dump(self.map_support_docs_titles,open("map_support_docs_titles.p","w"))
        pickle.dump(self.map_support_docs_super_titles,open("map_support_docs_super_titles.p","w"))
        pickle.dump(self.map_support_docs_title_subtitle,open("map_support_docs_title_subtitle.p","w"))
        pickle.dump(self.map_support_docs_text_list,open("map_support_docs_text_list.p","w"))
        pickle.dump(self.map_support_docs_text_sentences_list,open("map_support_docs_text_sentences_list.p","w"))
        pickle.dump(self.map_support_docs_complete_text,open("map_support_docs_complete_text.p","w"))
        pickle.dump(self.map_support_docs_complete_text_list,open("map_support_docs_complete_text_list.p","w"))
        pickle.dump(self.map_support_docs_exact_title,open("map_support_docs_exact_title.p","w"))
        pickle.dump(self.map_support_docs_exact_sub_title,open("map_support_docs_exact_sub_title.p","w"))
        pickle.dump(self.map_support_docs_paragraph_titles_bodies,open("map_support_docs_paragraph_titles_bodies.p","w"))
        pickle.dump(self.ignore_support_docs,open("handle_super_titles_ignore_support_docs","w"))

    def _create_all_words_set(self):
        all_words = set()
        all_support_doc_words_map = {}
        all_solved_question_words_map = {}
        all_support_doc_title_words_map = {}
        all_words_map = {}
        all_words_files = {}
        stop_words = list(stopwords.words('english'))
        try:
            considered_all_words = pickle.load(open("considered_all_words","r"))
            considered_all_solved_questions_words = pickle.load(open("considered_all_solved_questions_words","r"))
            considered_all_support_doc_words = pickle.load(open("considered_all_support_doc_words","r"))
            considered_all_support_doc_title_words = pickle.load(open("considered_all_support_doc_title_words","r"))
            considered_all_words_maps = pickle.load(open("considered_all_words_maps","r"))
            considered_all_words_files = pickle.load(open("considered_all_words_files","r"))
            return considered_all_words,considered_all_solved_questions_words,considered_all_support_doc_words,considered_all_support_doc_title_words,considered_all_words_maps,considered_all_words_files
        # get all words in super titles
        except IOError:
            print "Pakkav dictionaries"
        for doc in self.map_support_docs_super_titles.keys():
            super_title=self.map_support_docs_super_titles[doc]+self.map_support_docs_text[doc]
            for word in super_title.split(' '):
                if word in stop_words:
                    continue
                all_words.add(word)
                all_words_map[word] = all_words_map.setdefault(word,0) + 1
                all_support_doc_words_map[word] = all_support_doc_words_map.setdefault(word,0) + 1
                if word not in all_words_files:
                    all_words_files[word] = set(doc)
                else:
                    all_words_files[word].add(doc)

            for word in self.map_support_docs_super_titles[doc].split(' '):
                if word in stop_words:
                    continue
                all_support_doc_title_words_map[word] = all_support_doc_title_words_map.setdefault(word,0) + 1
            

        # get all words in question texts
        for i in range(len(self.solved_files_question_text)):
            text=self.solved_files_question_text[i]
            for word in text.split(' '):
                if word in stop_words:
                    continue
                all_words.add(word)
                all_words_map[word] = all_words_map.setdefault(word,0) + 1
                all_solved_question_words_map[word] = all_solved_question_words_map.setdefault(word,0) + 1
                if word not in all_words_files:
                    all_words_files[word] = set(self.solved_files[i])
                else:
                    all_words_files[word].add(self.solved_files[i])
        
        considered_all_words = []
        considered_all_words_maps = {}
        considered_all_support_doc_words = []
        considered_all_solved_questions_words = []
        considered_all_support_doc_title_words = []
        considered_all_words_files = {} 
        for word,count in all_words_map.items():
            if count>7:
                considered_all_words.append(word)
                considered_all_words_maps[word] = all_words_map[word]
                considered_all_words_files[word] = all_words_files[word]
                if word in all_support_doc_words_map:
                    considered_all_support_doc_words.append(word)
                if word in all_solved_question_words_map:
                    considered_all_solved_questions_words.append(word)
                if word in all_support_doc_title_words_map:
                    considered_all_support_doc_title_words.append(word)
        #all_words = list(all_words)
        print "Total dictionary size (discussions + support docs): %d"%len(considered_all_words)
        print "Support doc dictionary size %d and solved questions dictionary size %d"%(len(considered_all_support_doc_words),len(considered_all_solved_questions_words))
        pickle.dump(considered_all_words,open("considered_all_words","w"))
        pickle.dump(considered_all_solved_questions_words,open("considered_all_solved_questions_words","w"))
        pickle.dump(considered_all_support_doc_words,open("considered_all_support_doc_words","w"))
        pickle.dump(considered_all_support_doc_title_words,open("considered_all_support_doc_title_words","w"))
        pickle.dump(considered_all_words_maps,open("considered_all_words_maps","w"))
        pickle.dump(considered_all_words_files,open("considered_all_words_files","w"))
        return considered_all_words,considered_all_solved_questions_words,considered_all_support_doc_words,considered_all_support_doc_title_words,considered_all_words_maps,considered_all_words_files

    def _understand_support_doc_text(self):
        print "Understanding text from %d support docs" %len(self.all_support_docs)
        try:
            print "Aaha Darshu, short cut for understanding text"
            self.map_support_docs_text = pickle.load(open("map_support_docs_text.p","r"))
            self.ignore_support_docs = pickle.load(open("understand_text_ignore_docs","r"))
            return
        except IOError:
            print "Did not find map support docs text"
        for curr_doc in self.all_support_docs:
            #print curr_doc
            curr_doc_path = DOC_PATH+curr_doc
            soup=getSoup(curr_doc_path)
            #print soup
            if not soup:
                print curr_doc,"has no body"
                self.ignore_support_docs.add(curr_doc)
                continue
            #print soup
            doc_text = getText(soup)
            doc_text = str(doc_text)
            #print doc_text
            if not doc_text:
                self.ignore_support_docs.add(curr_doc)
                continue
            self.map_support_docs_text[curr_doc] = doc_text.lower()
        pickle.dump(self.map_support_docs_text,open("map_support_docs_text.p","w"))
        pickle.dump(self.ignore_support_docs,open("understand_text_ignore_docs","w"))

def create_w2v_text(bsl=None):

    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    
    w2v_file = open('w2vec.txt','w')
    sfqt = bsl.solved_files_question_text
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    for sentence in sfqt:
        sentence = sentence.lower()
        sentence = regex.sub(' ',sentence)
        sent_lst = sentence.split()
        for w in sent_lst: w2v_file.write(w+' ')#print w, 
    
    for support_doc in bsl.map_support_docs_text.keys():
        
        text = str(bsl.map_support_docs_text[support_doc])
        sentence = text
        sentence = sentence.lower()
        sentence = regex.sub(' ',sentence)
        sent_lst = sentence.split()
        for w in sent_lst: w2v_file.write(w+' ')#print w,
    w2v_file.close()

def lucene_text(bsl=None):
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    
    lucene_text_file = open('lucene_corpus.txt','w')
    for support_doc in bsl.map_support_docs_text.keys():
        text = str(bsl.map_support_docs_text[support_doc])
        sentence = text
        sentence = sentence.lower()
        sent_lst = sentence.split()
        #print support_doc+"\t"+' '.join(sent_lst)
        lucene_text_file.write(support_doc+"\t"+' '.join(sent_lst)+'\n')
    lucene_text_file.close()

def tao_model_toy_test_phrase_picker(bsl=None,use_sentences=False,toy_files=[]):
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    tao_model_text_file = open("tao_model_toy_addendum.txt","w")
    tao_test_file = open("test_toy.txt","w")
    support_doc_considered = set()
    if not toy_files:
        toy_files = ["../DATA/7622960","../DATA/7622175","../DATA/7622066","../DATA/7621761","../DATA/7621729"]
    for solved_file in toy_files:
        #support_doc = bsl.solved_files_support_docs[solved_file]
        #support_doc_text_list = bsl.map_support_docs_text_list[support_doc]
        #candidate_files = []
        positive_candidates = []
        negative_candidates = []
        for support_doc in bsl.all_support_docs:
            if support_doc != bsl.solved_files_support_docs[solved_file]:
                continue
            #support_doc_text_list = bsl.map_support_docs_text_list[support_doc]
            support_doc_para_titles = bsl.map_support_docs_paragraph_titles_bodies[support_doc][0]
            if len(support_doc_para_titles) == 0:
                print "%s has not paragraph titles" %support_doc
            support_doc_para_bodies = bsl.map_support_docs_paragraph_titles_bodies[support_doc][1]
            title = str(bsl.map_support_docs_title_subtitle[support_doc]).lower()
            candidate_files = []
            for i in range(len(support_doc_para_titles)):
                text = ""
                para_title = title
                for j in range(i,i+1):
                    text = text+" "+support_doc_para_bodies[j].lower()
                    para_title = para_title+" "+support_doc_para_titles[j].lower()
                    support_doc_broken = support_doc+"_"+str(i)+"_"+str(j)
                    if support_doc_broken not in support_doc_considered:
                        tao_model_text_file.write(support_doc_broken+"\t"+para_title.strip()+"\t"+text.strip()+"\n")
                        support_doc_considered.add(support_doc_broken)
                    candidate_files.append(support_doc_broken)
            #tao_model_text_file.write(support_doc+"\t"+title+"\t"+' '.join(support_doc_text_list).lower().strip()+"\n")
            candidate_files.append(support_doc)
            if support_doc == bsl.solved_files_support_docs[solved_file]:
                positive_candidates = candidate_files
            else:
                negative_candidates = negative_candidates + candidate_files
        #tao_test_file.write(solved_file[solved_file.rfind('/')+1:]+"\t"+(' '.join(positive_candidates).strip())+"\t"+(' '.join(positive_candidates+negative_candidates).strip())+"\n")
            tao_test_file.write(solved_file[solved_file.rfind('/')+1:]+"\t"+(' '.join([support_doc]).strip())+"\t"+(' '.join(list(set(candidate_files)-set(support_doc))).strip())+"\n")
    print "Size of all split guys %d" %(len(positive_candidates+negative_candidates))
    tao_model_text_file.close()
    tao_test_file.close()

def tao_model_best_phrase(bsl=None,use_sentences=False,use_tfidf=False):

    f_name = "map_file_bow_%d_nonbinary.pkl" % len(bsl.solved_files)
    if use_tfidf:
        f_name = "map_file_bow_%d_tfidf.pkl" % len(bsl.solved_files)
    with open(f_name, "r") as f:
        map_file_bow = pickle.load(f)
    print "Picked up a map of size %d" %(len(map_file_bow))
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    file_name = "paragraph"
    if use_sentences:
        file_name = "sentence"
    start_from = 0
    end_from = 919
    cosine_avg = 0.0
    tao_model_text_file = open(str(start_from)+'_'+str(end_from)+'_tao_model_best_'+file_name+'s.txt','w')
    support_doc_question_file = open(str(start_from)+'_'+str(end_from)+'_support_doc_question'+file_name+'.txt','w')
    considered_cos = 0
    for J,support_doc in enumerate(bsl.all_support_docs[start_from:end_from]):
        I = J + start_from
        print "Considering support doc number %d" %(I+1)
        label = str(bsl.map_support_docs_title_subtitle[support_doc]).lower()
        label = ' '.join(label.split()).strip()
        #support_doc_text_list = bsl.map_support_docs_text_list[support_doc]
        support_doc_paras = bsl.map_support_docs_paragraph_titles_bodies[support_doc]
        support_doc_text_list = []
        for i in range(len(support_doc_paras)):
            support_doc_text_list.append((support_doc_paras[0][i].strip()+" "+support_doc_paras[1][i].strip()).strip())
        if use_sentences:
            support_doc_text_list = bsl.map_support_docs_complete_text_list[support_doc]
        support_doc_bow_list = []
        for i in range(len(support_doc_text_list)):
            support_doc_bow_list.append(bow_vector(support_doc_text_list[i].lower().strip().split(),bsl.all_words,binary=False))
            #if use_tfidf:
            #    support_doc_bow_list[-1] = tfidf_bow_vector(support_doc_text_list[i].lower().strip().split(),bsl.all_words,bsl.all_words_files)
        pairs_used = {}
        ctr = 0
        for i in range(len(bsl.solved_files)):
            #if bsl.solved_files_support_docs[bsl.solved_files[i]] != support_doc:
            #    continue
            start_end_list,text,current_cosine = get_best_sentences(i,support_doc,support_doc_bow_list,bsl,map_file_bow,use_sentences,use_tfidf)
            if current_cosine <0:
                print "fuck"
                print support_doc,i
            cosine_avg += current_cosine
            print "Cosine %f, %f" %(cosine_avg,current_cosine)
            considered_cos += 1
            array_select = [0 for _ in range(len(support_doc_bow_list))]
            for K in range(len(start_end_list)):
                for k in range(start_end_list[K][0],start_end_list[K][1]):
                    array_select[k] = 1
            #for k in range(s2,e2):
            #    array_select[k] = 1
            #for k in range(s3,e3):
            #    array_select[k] = 1
            if tuple(array_select) not in pairs_used:
                ctr += 1
                pairs_used[tuple(array_select)] = ctr
                tao_model_text_file.write(support_doc+'_'+str(pairs_used[tuple(array_select)])+"\t"+label+"\t"+text+"\n")
            print "Support doc %d, question %d" %(I+1,i+1)
            support_doc_question_file.write(support_doc+'_'+str(pairs_used[tuple(array_select)])+'\t'+bsl.solved_files[i]+'\n')
            #if tuple([s,e,s2,e2]) not in pairs_used:
            #    tao_model_text_file.write(support_doc+'_'+str(s)+'_'+str(e)+'_'+str(s2)+'_'+str(e2)+"\t"+label+"\t"+text+"\n")
            #    pairs_used.add(tuple([s,e,s2,e2]))
            #print "Support doc %d, question %d" %(I+1,i+1)
            #support_doc_question_file.write(support_doc+'_'+str(s)+'_'+str(e)+'_'+str(s2)+'_'+str(e2)+'\t'+bsl.solved_files[i]+'\n')
    
    for i,solved_file in enumerate(bsl.solved_files):
        solved_file = solved_file[solved_file.rfind('/')+1:]
        label = bsl.solved_files_question_label[i]
        content = bsl.solved_files_question_content[i]

        #print solved_file+"\t"+label+"\t"+content
        tao_model_text_file.write(solved_file+"\t"+label+"\t"+content+'\n')

    tao_model_text_file.close()
    support_doc_question_file.close()
    print "Cosine avg %f" %(cosine_avg/(considered_cos))
    print "Considered cosine %d" %(considered_cos)

def tao_model_text(bsl=None):
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    tao_model_text_file = open('tao_model.txt','w')
    for support_doc in bsl.map_support_docs_text.keys():
        label = str(bsl.map_support_docs_title_subtitle[support_doc])
        label = label.lower()
        label_lst = label.split()
        label = ' '.join(label_lst)
       
        #text = str(' '.join(bsl.map_support_docs_text_list[support_doc])).strip().lower()
        #text = str(bsl.map_support_docs_complete_text[support_doc]).lower()
        #sentence = text
        #sentence = sentence.lower()
        #sent_lst = sentence.split()
        #text = ' '.join(sent_lst)
        text = bsl.map_support_docs_text[support_doc].strip().lower()
        
        #print support_doc+"\t"+label+"\t"+text
        tao_model_text_file.write(support_doc+"\t"+label+"\t"+text+'\n')
    
    for i,solved_file in enumerate(bsl.solved_files):
        solved_file = solved_file[solved_file.rfind('/')+1:]
        label = bsl.solved_files_question_label[i]
        content = bsl.solved_files_question_content[i]
        
        #print solved_file+"\t"+label+"\t"+content
        tao_model_text_file.write(solved_file+"\t"+label+"\t"+content+'\n')
    tao_model_text_file.close()

def tao_train_data_pairs(bsl=None,use_para=False,use_sentences=True,ignore_files=[]):
    
    if use_para:
        print "Using paragraph based support documents"
    map_solved_file = {}
    support_docs_para = []
    if use_para:
        support_doc_question_file_name = "support_doc_questionparagraph.txt"
        if use_sentences:
            support_doc_question_file_name = "support_doc_questionsentence.txt"
        support_doc_question_file = open(support_doc_question_file_name,"r")
        support_doc_lines = support_doc_question_file.readlines()
        support_doc_question_file.close()
        for line in support_doc_lines:
            line = line.strip()
            line_list = line.split('\t')
            if line_list[1] not in map_solved_file:
                map_solved_file[line_list[1]] = [line_list[0]]
            else:
                map_solved_file[line_list[1]].append(line_list[0])
        print "Created solved file map based on paragraph based support document"
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    import random
    no_candidates = 10000

    train_file_name = 'train_para.txt'
    if use_sentences:
        train_file_name = 'train_sentences.txt'
    if not use_para:
        train_file_name = 'train.txt'    
    train_file = open(train_file_name,'w')

    for i,solved_file in enumerate(bsl.solved_files):
        #if solved_file in bsl.solved_files_with_support_docs:
        #    continue
        if solved_file in ignore_files:
            continue
        cor_doc = bsl.solved_files_support_docs[solved_file]
        if use_para:
            support_docs_para = map_solved_file[solved_file]
            for sup_doc_para in support_docs_para:
                if cor_doc+'_' in sup_doc_para:
                    cor_doc = sup_doc_para
                    break
        solved_file = solved_file[solved_file.rfind('/')+1:]
        start_ind = random.randint(0,max(0,len(bsl.all_support_docs)-no_candidates))
        negative_examples = ""
        for neg_docs in bsl.all_support_docs[start_ind:start_ind+no_candidates]:
            if neg_docs==cor_doc:
                continue
            if use_para:
                for sup_doc_para in support_docs_para:
                    if neg_docs+'_' in sup_doc_para:
                        neg_docs = sup_doc_para
                        break
            negative_examples = neg_docs+" "+negative_examples
       # print solved_file+"\t"+cor_doc+"\t"+negative_examples
        train_file.write(solved_file+"\t"+cor_doc+"\t"+negative_examples+'\n')
    train_file.close()

def get_best_sentences(question_indx, support_doc, support_doc_bow_list=[], bsl=None, map_file_bow={}, use_sentences=False, use_tfidf=False):
    print "Using tf-idf?",use_tfidf
    if len(map_file_bow.keys()) == 0:
        f_name = "map_file_bow_%d_nonbinary.pkl" % len(bsl.solved_files)
        print "Loading map file bow"
        if use_tfidf:
            f_name = "map_file_bow_%d_tfidf.pkl" % len(bsl.solved_files)
            print "Using tf-idf weighted cosine similarity"
        with open(f_name, "r") as f:
            map_file_bow = pickle.load(f)
        print "Received a map of size %d" %(len(map_file_bow.keys()))
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    print "Finding question index for %s" %question_indx
    #f_error = open("error.txt","a")
    if type(question_indx) != int:
        question_indx = bsl.solved_files.index(question_indx)
    question_bow = map_file_bow[bsl.solved_files[question_indx]]
    #question_bow = bow_vector(bsl.solved_files_question_text[question_indx].lower().split(),bsl.all_words,binary=False)
    #important_ques = ""
    #for word in bsl.solved_files_question_text[question_indx].lower().split():
    #    if word in bsl.all_words and question_bow[bsl.all_words.index(word)]>0:
    #        important_ques = important_ques+word+" "
    #f_error.write(important_ques+"\n")
    start_ind = 0
    end_ind = 0
    cosine_max = 0.0
    best_subtext = []
    pair_scores = {}
    start_end_list = []
    top_n = 1
    support_doc_text_list = bsl.map_support_docs_text_list[support_doc]
    if use_sentences:
        support_doc_text_list = bsl.map_support_docs_complete_text_list[support_doc]
    if len(support_doc_bow_list) == 0:
        for i in range(len(support_doc_text_list)):
            support_doc_bow_list.append(bow_vector(support_doc_text_list[i].lower().strip().split(),bsl.all_words,binary=False))
            if use_tfidf:
                support_doc_bow_list[-1] = tfidf_bow_vector(support_doc_text_list[i].lower().strip().split(),bsl.all_words,bsl.all_words_files)
    outer_loop = len(support_doc_bow_list)
    if use_sentences:
        outer_loop -= 1
    for i in range(outer_loop):
        use_subtext_end = i+1
        use_subtext_start = i
        if use_sentences:
             use_subtext_start = i+1
             use_subtext_end = min(len(support_doc_bow_list),i+6)
        support_doc_bow = np.copy(support_doc_bow_list[i])
        for j in range(use_subtext_start,use_subtext_end):
            support_doc_bow += support_doc_bow_list[j]
            if np.linalg.norm(question_bow) == 0.0 or np.linalg.norm(support_doc_bow) ==0.0:
                current_cosine = 0.0
            else:
                current_cosine = np.dot(question_bow,support_doc_bow)/(np.linalg.norm(question_bow)*np.linalg.norm(support_doc_bow))
            pair_scores[tuple([i,j+1])] = current_cosine
            #f_error.write("%s" %(support_doc_text_list[i])+"\n")
            #f_error.write("%d %f" %(j+1,current_cosine)+"\n")
            #important_text = ""
            #for word in support_doc_text_list[i].lower().split():
            #    if word in bsl.all_words and support_doc_bow_list[i][bsl.all_words.index(word)]>0:
            #        important_text = important_text+word+" "
            #f_error.write(important_text+"\n")
            if current_cosine > cosine_max:
                cosine_max = current_cosine
                start_ind = i
                end_ind = j+1
        support_doc_bow = []
    start_ind_2 = 0
    end_ind_2 = 0
    start_ind_3 = 0
    end_ind_3 = 0
    for i,w in enumerate(sorted(pair_scores, key=pair_scores.get, reverse=True)):
        if i<top_n:
            start_end_list.append(tuple([w[0],w[1]]))
        #if i==0:
        #    start_ind = w[0]
        #    end_ind = w[1]
        #elif i==1:
        #    start_ind_2 = w[0]
        #    end_ind_2 = w[1]
        #elif i==2:
        #    start_ind_3 = w[0]
        #    end_ind_3 = w[1]
        #else:
        #    break
    array_select = [0 for _ in range(len(support_doc_bow_list))]
    for I in range(len(start_end_list)):
        for i in range(start_end_list[I][0],start_end_list[I][1]):
            array_select[i] = 1
    #for i in range(start_ind,end_ind):
    #    if i >= len(array_select):
    #        continue
    #    array_select[i] = 1
    #for i in range(start_ind_2,end_ind_2):
    #    if i >= len(array_select):
    #        continue
    #    array_select[i] = 1
    #for i in range(start_ind_3,end_ind_3):
    #    if i >= len(array_select):
    #        continue
    #    array_select[i] = 1
    for i in range(len(array_select)):
        if array_select[i]==1:
            best_subtext.append(support_doc_text_list[i])
    return start_end_list,' '.join(best_subtext).strip(),cosine_max
    #f_error.close()

def tao_test_data_pairs(bsl=None,use_para=False,use_sentences=False,ignore_files=[],combine_documents=False): 
    
    map_solved_file = {}
    support_docs_para = []
    if use_para:
        support_doc_question_file_name = "support_doc_questionparagraph.txt"
        if use_sentences:
            support_doc_question_file_name = "support_doc_questionsentence.txt"
        support_doc_question_file = open(support_doc_question_file_name,"r")
        support_doc_lines = support_doc_question_file.readlines()
        support_doc_question_file.close()
        for line in support_doc_lines:
            line = line.strip()
            line_list = line.split('\t')
            if line_list[1] not in map_solved_file:
                map_solved_file[line_list[1]] = [line_list[0]]
            else:
                map_solved_file[line_list[1]].append(line_list[0])

    if not bsl:
        bsl = Baseline(sys.maxint) 
        bsl.populate_data() 
    import random 
    no_candidates = 999999
    test_file_name = 'test_para.txt'
    if use_sentences:
        test_file_name = 'test_sentences.txt'
    if not use_para:
        test_file_name = 'test.txt'

    support_doc_pairs = {}#pickle.load( open( "support_doc_pairs.p", "rb" ) )

    test_file = open(test_file_name,'w')
    for i,solved_file in enumerate(bsl.solved_files):
        #if solved_file not in bsl.solved_files_with_support_docs:
        #    continue
        if solved_file in ignore_files:
            continue
        cor_doc = bsl.solved_files_support_docs[solved_file]
        cor_doc_list = [cor_doc]
        if cor_doc in support_doc_pairs and combine_documents:
            cor_doc_list += support_doc_pairs[cor_doc]
        cor_doc_set = set(cor_doc_list)
        if use_para:
            support_docs_para = map_solved_file[solved_file]
            for sup_doc_para in support_docs_para:
                if cor_doc+'_' in sup_doc_para:
                    cor_doc = sup_doc_para
                    break
        solved_file = solved_file[solved_file.rfind('/')+1:] 
        start_ind = random.randint(0,max(len(bsl.all_support_docs)-no_candidates,0))
        negative_examples = "" 
        for neg_docs in bsl.all_support_docs[start_ind:start_ind+no_candidates]: 
            if neg_docs in cor_doc_set:
                continue
            if use_para:
                for sup_doc_para in support_docs_para:
                    if neg_docs+'_' in sup_doc_para:
                        neg_docs = sup_doc_para
                        break
            negative_examples = neg_docs+" "+negative_examples 
        test_file.write(solved_file+"\t"+(' '.join(cor_doc_list)).strip()+"\t"+(' '.join(cor_doc_list).strip()+" "+negative_examples).strip()+'\n')
    test_file.close()

def create_parallel_data(bsl=None):
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    
    f_ques = open('questions.txt','w')
    f_ans = open('answers.txt','w')
    for i,solved_file in enumerate(bsl.solved_files):
        cor_doc = bsl.solved_files_support_docs[solved_file]
        question = bsl.solved_files_question_content[i]
        question_parsed = ""
        ctr = 20
        for q in question.split():
            if q in bsl.all_words:
                ctr = ctr-1
                if ctr==0:
                    break
                question_parsed = question_parsed+" "+q
        question_parsed = question_parsed.strip()
        f_ques.write(question_parsed+'\n')
        answer = bsl.map_support_docs_titles[cor_doc]
        answer_parsed = ""
        ctr = 20
        for a in answer.split():
            if a in bsl.all_words:
                ctr = ctr-1
                if ctr==0:
                    break
                answer_parsed = answer_parsed+" "+a
        answer_parsed = answer_parsed.strip()
        f_ans.write(answer_parsed+'\n')

    f_ques.close()
    f_ans.close()

def kfold(bsl=None,use_para=False,use_sentences=False,fold = 0):
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    
    fold = 0
    train_ignore = []
    test_ignore = []
    for i,solved_file in enumerate(bsl.solved_files):
        if solved_file in bsl.solved_files_with_support_docs:
            if i%5 == fold:
                train_ignore.append(solved_file)
            else:
                test_ignore.append(solved_file)
        else:
            test_ignore.append(solved_file)
    tao_train_data_pairs(bsl,use_para,use_sentences,train_ignore)
    tao_test_data_pairs(bsl,use_para,use_sentences,test_ignore)
            

def copy_relevant_files(bsl=None):
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    from shutil import copyfile 
    mypath = '../DATA_short/'
    if not os.path.exists(mypath):
            os.makedirs(mypath)

    for solved_file in bsl.solved_files:
        copyfile(solved_file,mypath+solved_file[solved_file.rfind('/')+1:])

def get_dissimilar_words(bsl):
    
    model = gensim.models.Word2Vec.load_word2vec_format('apple-text-vector.bin', binary=True)
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    support_doc_words = set(bsl.all_support_docs_words)
    support_doc_title_words = set(bsl.all_support_docs_title_words)
    dissimilar_words = {}
    count_similar_not_found = 0
    out_of_dict_word = 0
    for word in bsl.all_solved_question_words:
        if word not in support_doc_words:
            dissimilar_words[word] = ""
            if word not in model:
                out_of_dict_word += 1
                continue
            top_cosine_similar = model.most_similar(positive = [word],topn=50)
            for w,cos in top_cosine_similar:
                if w in support_doc_title_words:
                    dissimilar_words[word] = w
                    break
            if dissimilar_words[word]:
                continue
            for w,cos in top_cosine_similar:
                if w in support_doc_words:
                    dissimilar_words[word] = w
                    break
            if not dissimilar_words[word]:
                count_similar_not_found += 1
    print "%d words without a partner, %d out of dict from %d" %(count_similar_not_found,out_of_dict_word,len(dissimilar_words.keys())-out_of_dict_word)
    return dissimilar_words

def get_support_doc_cosine_sim(bsl=None):

    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()

    support_doc_bow_vectors = []
    for support_doc in bsl.all_support_docs:
        support_doc_super_title = bsl.map_support_docs_text[support_doc]
        support_doc_bow_vectors.append(bow_vector(support_doc_super_title.split(),bsl.all_words,binary=False))
    file_cosine_support_doc = open('support_doc_cosine.txt','w')
    for i,support_doc_i in enumerate(bsl.all_support_docs):
        for j,support_doc_j in enumerate(bsl.all_support_docs):
            if i >= j:
                continue
            file_cosine_support_doc.write(support_doc_i+" "+support_doc_j+" "+str(np.dot(support_doc_bow_vectors[i],support_doc_bow_vectors[j])/(np.linalg.norm(support_doc_bow_vectors[i])*np.linalg.norm(support_doc_bow_vectors[j])))+"\n")
    file_cosine_support_doc.close() 

def get_lda_classes(bsl=None):

    p_stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    texts = []
    important_word_counts = {}
    doc_set = []
    for support_doc in bsl.all_support_docs:
        doc_set.append(bsl.map_support_docs_text[support_doc])
    for i in doc_set:
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [i for i in tokens if not i in en_stop and i.isalpha()]
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        texts.append(stemmed_tokens)
        for j in stemmed_tokens:
            if j in important_word_counts:
                important_word_counts[j]+=1.0
            else:
                important_word_counts[j]=1.0
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=20)
        file_lda = open('support_doc_lda.txt','w')
        for i in range(len(corpus)):
            topic = ldamodel[corpus[i]][0][0]
            file_lda.write(bsl.all_support_docs[i]+" "+str(topic)+"\n")
        file_lda.close()

def create_toy_data(bsl=None):
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    #tao_model_toy_test_phrase_picker(bsl)
    #toy_files = ["../DATA/7622960","../DATA/7622175","../DATA/7622066","../DATA/7621761","../DATA/7621729"]
    toy_files = bsl.solved_files_with_support_docs
    tao_model_toy_test_phrase_picker(bsl,use_sentences=False,toy_files=toy_files)
    tao_model_text(bsl)
    tao_train_data_pairs(bsl,use_para=False,use_sentences=False,ignore_files=toy_files)

def create_basic_data(bsl=None):
    
    print "Welcome to data creation"
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    tao_model_text(bsl)
    train_ignore = []
    for i,s_f in enumerate(bsl.solved_files):
        if (i+1)%3 == 0:
            train_ignore.append(s_f)
    test_ignore = list(set(bsl.solved_files) - set(train_ignore))
    print "Number of train files %d" %len(test_ignore)
    print "Number of test files %d" %len(train_ignore)
    tao_train_data_pairs(bsl,use_para=False,use_sentences=False,ignore_files=train_ignore)
    tao_test_data_pairs(bsl,use_para=False,use_sentences=False,ignore_files=test_ignore)

def train_encoding_data(bsl=None):
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    toy_files = bsl.solved_files
    tao_model_text(bsl)
    tao_model_toy_test_phrase_picker(bsl,use_sentences=False,toy_files=toy_files)
    tao_train_data_pairs(bsl,use_para=False,use_sentences=False,ignore_files=bsl.solved_files_with_support_docs)

def test_cosine(bsl=None):
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    map_ids = pickle.load( open( "ids_encodign.p", "rb" ) )
    test_file = open('test_enc.txt','w')
    train_file = open('train_enc.txt','w')
    for solved_file in bsl.solved_files:
        isTrain = True
        if solved_file in bsl.solved_files_with_support_docs:
            isTrain = False
        solved_file_id = solved_file[solved_file.rfind('/')+1:]
        if solved_file_id not in map_ids:
            continue
        correct_support_doc = ""
        candidates = []
        for support_doc in bsl.all_support_docs:
            para_num = 0
            best_para = -1
            best_cosine = np.dot(map_ids[support_doc],map_ids[solved_file_id])
            while True:
                if support_doc+"_"+str(para_num)+"_"+str(para_num) in map_ids:
                    current_cosine = np.dot(map_ids[support_doc+"_"+str(para_num)+"_"+str(para_num)],map_ids[solved_file_id])
                    if current_cosine > best_cosine:
                        best_para = para_num
                        best_cosine = current_cosine
                    para_num += 1
                else:
                    break
            support_doc_chosen = support_doc
            if best_para != -1:
                support_doc_chosen = support_doc+"_"+str(best_para)+"_"+str(best_para)
            candidates.append(support_doc_chosen)
            if support_doc == bsl.solved_files_support_docs[solved_file]:
                correct_support_doc = support_doc_chosen
                if isTrain:
                    candidates.remove(correct_support_doc)
        if not isTrain: 
            test_file.write(solved_file_id+"\t"+correct_support_doc+"\t"+(' '.join(candidates)).strip()+"\n")
        else:
            train_file.write(solved_file_id+"\t"+correct_support_doc+"\t"+(' '.join(candidates)).strip()+"\n")
    test_file.close()
    train_file.close()

def create_encoding_data(bsl):

    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    train_encoding_data(bsl)
    test_cosine(bsl)    

def top_cosine(bsl=None,top_n=919):
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()

    map_ids = pickle.load( open( "ids_encodign.p", "rb" ) )
    test_file = open("test_top_encoding.txt","w")
    successful = 0
    for solved_file in bsl.solved_files_with_support_docs:
        solved_file_id = solved_file[solved_file.rfind('/')+1:]
        best_candidates = {}
        positive_candidates = []
        for support_doc in bsl.all_support_docs:
            if support_doc == bsl.solved_files_support_docs[solved_file]:
                positive_candidates.append(support_doc)
                para_num = 0
                while True:
                    considered_support_doc = support_doc+"_"+str(para_num)+"_"+str(para_num)
                    if considered_support_doc not in map_ids:
                        break
                    positive_candidates.append(considered_support_doc)
                    para_num += 1
            para_num = 0
            best_candidates[support_doc] = np.dot(map_ids[solved_file_id],map_ids[support_doc])
            while True:
                considered_support_doc = support_doc+"_"+str(para_num)+"_"+str(para_num)
                if considered_support_doc not in map_ids:
                    break
                best_candidates[considered_support_doc] = np.dot(map_ids[considered_support_doc],map_ids[solved_file_id])
                para_num += 1
        top_support_docs = set(sorted(best_candidates, key=best_candidates.get, reverse=True)[:top_n])

        positive_candidates_picked = []
        for candidate in positive_candidates:
            if candidate in top_support_docs:
                positive_candidates_picked.append(candidate)

        if len(positive_candidates_picked) > 0:
            successful += 1
            test_file.write(solved_file_id+"\t"+(' '.join(positive_candidates_picked)).strip()+"\t"+(' '.join(top_support_docs)).strip()+"\n")
    print "A total of %d successful for a top %d threshold" %(successful,top_n)
    test_file.close()

def create_data_model(bsl):
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    tao_model_text(bsl)
    tao_model_toy_test_phrase_picker(bsl,toy_files=bsl.solved_files)

def getSimilarPairs(bsl):

    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    
    support_doc_pairs = pickle.load( open( "support_doc_pairs.p", "rb" ) )
    for support_doc in bsl.all_support_docs:
        if support_doc in support_doc_pairs:
            print
            print support_doc
            for s in support_doc_pairs[support_doc]:
                print s,

def getSupportDocsCosineSimilarities(bsl):

    dict_top_pairs = {}
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    map_ids = pickle.load( open( "ids_encodign.p", "rb" ) )
    map_support_doc_pairs = {}
    for i,support_doc_i in enumerate(bsl.all_support_docs):
        for j,support_doc_j in enumerate(bsl.all_support_docs):
            if j <= i:
                continue
            map_support_doc_pairs[tuple([support_doc_i,support_doc_j])] = np.dot(map_ids[support_doc_i],map_ids[support_doc_j])
    file_pairs = open("support_doc_pairs.txt","w")
    for w in sorted(map_support_doc_pairs, key=map_support_doc_pairs.get, reverse=True):
        file_pairs.write(w[0]+"-"+w[1]+" "+str(map_support_doc_pairs[w])+'\n')
        if map_support_doc_pairs[w] > 0.8:
            if w[0] in dict_top_pairs:
                dict_top_pairs[w[0]].append(w[1])
            else:
                dict_top_pairs[w[0]] = [w[1]]
            if w[1] in dict_top_pairs:
                dict_top_pairs[w[1]].append(w[0])
            else:
                dict_top_pairs[w[1]] = [w[0]]
    file_pairs.close()
    pickle.dump( dict_top_pairs, open( "support_doc_pairs.p", "wb" ) )


def getUniqSupportDocs(bsl):

    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    ctr = 0
    for support_doc in bsl.all_support_docs:
        soup = getSoup(DOC_PATH+support_doc)
        if getDocName(soup) == support_doc:
            ctr += 1
    print "%d unique support docs" %ctr

def supportDocsUniqSubtitles(bsl=None):
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    uniq_subtitles = set()
    support_doc_common = {}
    for support_doc in bsl.all_support_docs:
        uniq_subtitles.add(bsl.map_support_docs_exact_sub_title[support_doc])
    
    print "%d number of uniq support docs is now" %(len(uniq_subtitles))

def cluster_text(bsl=None,num_topics=4000,use_answers=True,combine_question_answers=False,forTest=True):
    
    p_stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')

    print "Beginning clustering for %d topics" %num_topics    

    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    
    doc_set = []
    texts = []
    considered_files = []
    
    assert len(bsl.solved_files) == len(bsl.solved_files_question_answer)
    for i,solved_file in enumerate(bsl.solved_files):
        if forTest and i%500 == 0:
            continue
        answer_text = bsl.solved_files_question_answer[i]+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[solved_file]]
        if not use_answers and not combine_question_answers:
            answer_text = bsl.solved_files_question_text[i]
        if combine_question_answers:
            answer_text += " "+bsl.solved_files_question_text[i]
        doc_set.append(answer_text)
        considered_files.append(solved_file)

    for i in doc_set:
        raw = i.lower().strip()
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [i for i in tokens if not i in en_stop and i.isalpha()]
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        texts.append(stemmed_tokens)
        
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=20, minimum_probability=0.0)
    print "LDA model created"
    topics_answer_lists = {}
    topics_counts = {}
    id_topics = {}
    for i in range(len(corpus)):
        topics = ldamodel[corpus[i]]
        scores = []
        for topic,value in topics:
            scores.append(value)
        max_ind = np.array(scores).argmax()
        id_topics[considered_files[i]] = scores
        if max_ind not in topics_answer_lists:
            topics_answer_lists[max_ind] = [considered_files[i]]
            topics_counts[max_ind] = 1
        else:
            topics_answer_lists[max_ind].append(considered_files[i])
            topics_counts[max_ind] += 1

    if forTest:
        doc_set_test = []
        texts_test = []
        considered_files_test = []
        for i,solved_file in enumerate(bsl.solved_files):
            if i%500 != 0:
                continue
            answer_text = bsl.solved_files_question_answer[i]+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[solved_file]]
            if not use_answers and not combine_question_answers:
                answer_text = bsl.solved_files_question_text[i]
            if combine_question_answers:
                answer_text += " "+bsl.solved_files_question_text[i]
            doc_set_test.append(answer_text)
            considered_files_test.append(solved_file)
        
        for i in doc_set_test:
            raw = i.lower().strip()
            tokens = tokenizer.tokenize(raw)
            stopped_tokens = [i for i in tokens if not i in en_stop and i.isalpha()]
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            texts_test.append(stemmed_tokens)
    
        dictionary_test = corpora.Dictionary(texts_test)
        corpus_test = [dictionary_test.doc2bow(text) for text in texts_test]

        test_id_topics = {}
        for i in range(len(corpus_test)):
            topics = ldamodel[corpus_test[i]]
            scores = []
            for topic,value in topics:
                scores.append(value)
            test_id_topics[considered_files_test[i]] = scores
        if use_answers:
            map_dump = open("test_answer_topics.p","w")
            pickle.dump(test_id_topics,map_dump)
            map_dump.close()
        elif forTest:
            map_dump = open("test_question_topics.p","w")
            pickle.dump(test_id_topics,map_dump)
            map_dump.close()

    print "Number of topics found %d" %len(topics_answer_lists)
    fileName = "answer_clusters.txt"
    if not use_answers:
        fileName = "question_clusters.txt"
    f_answer_clusters = open(fileName,"w")
    for topic in sorted(topics_counts, key=topics_counts.get, reverse=True):
        f_answer_clusters.write("Topic Number: %d Topic count: %d\n" %(topic,topics_counts[topic]))
        for answer in topics_answer_lists[topic]:
            index = bsl.solved_files.index(answer)
            question = bsl.solved_files_question_text[index]
            answer = bsl.solved_files_question_answer[index]+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[bsl.solved_files[index]]]
            question_id = bsl.solved_files[index]
            url = "https://discussions.apple.com/thread/"+question_id[question_id.rfind('/')+1:]
            f_answer_clusters.write(url+"\nQuestion:\n"+question+"\nAnswer:\n"+answer+"\n")
    f_answer_clusters.close()

    if use_answers:
        map_dump = open("answer_topics.p","w")
        pickle.dump(id_topics,map_dump)
        map_dump.close()
    else:
        map_dump = open("question_topics.p","w")
        pickle.dump(id_topics,map_dump)
        map_dump.close()


def tao_question_answer(bsl=None):
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()

    f_text = open("tao_model.txt","w")
    ignore_files = set()
    for i,solved_file in enumerate(bsl.solved_files):
        question_title = bsl.solved_files_question_label[i].lower().strip()
        question_content = bsl.solved_files_question_content[i].lower().strip()
        question_id = solved_file[solved_file.rfind('/')+1:]
        if question_title == "":
            ignore_files.add(i)
            continue
        f_text.write(question_id+"\t"+question_title+"\t"+question_content+"\n")
        answer_id = question_id+"_ans"
        total_answer = bsl.solved_files_question_answer[i].lower().strip()+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[solved_file]]
        answer_title = ' '.join(total_answer.strip().split()[:100])
        answer_text = total_answer.lower().strip()
        f_text.write(answer_id+"\t"+answer_title+"\t"+answer_text+"\n")

    f_text.close()
    f_train = open("train.txt","w")
    f_test = open("test.txt","w")
    answer_topics = pickle.load(open("question_topics.p","r"))
    question_clusters = {}
    cluster_questions = {}
    for i,solved_file in enumerate(bsl.solved_files):
         if i in ignore_files:
             continue
         topics = answer_topics[solved_file]
         topic_num = np.array(topics).argmax()
         if topic_num not in cluster_questions:
             cluster_questions[topic_num] = [solved_file]
         else:
             cluster_questions[topic_num].append(solved_file)
         question_clusters[solved_file] = topic_num

    for i,solved_file in enumerate(bsl.solved_files):
        if solved_file not in question_clusters:
            continue
        similar_files = [solved_file]#cluster_questions[question_clusters[solved_file]]
        import random
        candidates = []
        for cluster_id in cluster_questions:
            if cluster_id !=question_clusters[solved_file]:
                candidates.append(random.choice(cluster_questions[cluster_id]))
        pos = []
        neg = []
        for sim in similar_files:
            sim = sim[sim.rfind('/')+1:]
            pos.append(sim+"_ans")
        for cand in candidates:
            cand = cand[cand.rfind('/')+1:]
            neg.append(cand+"_ans")
        ques_id = solved_file[solved_file.rfind('/')+1:]
        if i%50 == 0:
            f_test.write(ques_id+"\t"+(' '.join(pos)).strip()+"\t"+(' '.join(neg)).strip()+"\n")
        else:
            pos = [random.choice(pos)]
            f_train.write(ques_id+"\t"+(' '.join(pos)).strip()+"\t"+(' '.join(neg)).strip()+"\n")
    f_train.close()
    f_test.close()

def get_results(bsl=None):
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    question_answers = pickle.load(open("question_answer.p","r"))
    results = open("results.txt","w")
    for question in question_answers:
        predicted_answer = question_answers[question]
        results.write("Question url: https://discussions.apple.com/thread/%s\n" %question[question.rfind('/')+1:])
        question_ind = bsl.solved_files.index(question)
        results.write("Question:\n%s\n" %bsl.solved_files_question_text[question_ind])
        results.write("Actual Answer:\n%s\n" %bsl.solved_files_question_answer[question_ind])
        predicted_ind = bsl.solved_files.index(predicted_answer)
        results.write("Predicted Answer:\n%s\n" %bsl.solved_files_question_answer[predicted_ind])
    results.close() 

def encoding_data_kmeans(bsl=None):

    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()

    f_text = open("tao_model.txt","w")
    ignore_files = set()
    for i,solved_file in enumerate(bsl.solved_files):
        question_title = bsl.solved_files_question_label[i].lower().strip()
        question_content = bsl.solved_files_question_content[i].lower().strip()
        question_text = bsl.solved_files_question_text[i].lower().strip()
        question_id = solved_file[solved_file.rfind('/')+1:]
        if question_title == "":
            ignore_files.add(i)
            continue
        answer_id = question_id+"_ans"
        total_answer = bsl.solved_files_question_answer[i].lower().strip()+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[solved_file]]
        answer_title = ' '.join(total_answer.strip().split()[:100])
        answer_text = total_answer.lower().strip()
        f_text.write(question_id+"\t"+answer_title+"\t"+question_text+"\n")

    f_text.close()
    f_train = open("train.txt","w")
    
    for i,solved_file in enumerate(bsl.solved_files):
        if i in ignore_files:
            continue
        question_id = solved_file[solved_file.rfind('/')+1:]
        f_train.write(question_id+"\t"+question_id+"\t"+question_id+"\n")
    f_train.close()

#def clustering_based_on_encoding(bsl):
    

def kmeans_clustering_encoding(bsl,num_clusters=100):

    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    ids_encoding = pickle.load(open("ids_encoding.p","r"))
    question_answer_encoding = {}
    for name in ids_encoding:
        question_name = DATA_PATH+name
        assert question_name in bsl.solved_files
        question_answer_encoding[question_name] = ids_encoding[name]
    skm = SphericalKMeans(n_clusters=num_clusters)
    question_id = []
    question_data = []
    for solved_file in bsl.solved_files:
        if solved_file not in question_answer_encoding:
            continue
        question_id.append(solved_file)
        question_data.append(question_answer_encoding[solved_file])
    print "Length of encodings %d" %len(question_data[-1])
    skm.fit(question_data)
    question_topics = {}

    # For file output
    topic_questions = {}
    topic_counts = {} 

    for ind,label in enumerate(skm.labels_):
        topic = [0 for _ in range(num_clusters)]
        topic[label] = 1
        question = question_id[ind]
        question_topics[question] = topic
        if label not in topic_counts:
            topic_counts[label] = 1
            topic_questions[label] = [question]
        else:
            topic_counts[label] += 1
            topic_questions[label].append(question)

    dump_file = open("question_topics.p","w")
    pickle.dump(question_topics,dump_file)
    dump_file.close()

    f_clusters = open("kmeans_clusters.txt","w")
    for topic in sorted(topic_counts, key=topic_counts.get, reverse=True):
        questions = topic_questions[topic]
        #print topic,questions
        f_clusters.write("Cluster Number %d, Cluster Count %d\n" %(topic,topic_counts[topic]))
        for question in questions:
            index = bsl.solved_files.index(question)
            url = "https://discussions.apple.com/thread/"+question[question.rfind('/')+1:]
            query = bsl.solved_files_question_text[index]
            answer = bsl.solved_files_question_answer[index]+"\nSupport Document: "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[question]]
            f_clusters.write("Question: %s\nQuery: %s\nAnswer: %s\n\n" %(url,query,answer))
    f_clusters.close()

def auto_encoding_clustering(bsl=None):
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    ignore_files = set()
    question_answer_encodings = pickle.load(open("ids_encoding.p","r"))
    individual_pair_score = {}
    out_of_language = 0

    question_correct = pickle.load(open("question_correct.p","r"))
    question_wrong = pickle.load(open("question_wrong.p","r"))
    clusters = pickle.load(open("clusters.p","r"))
    index_cluster = {}
    for label in clusters:
        for index in clusters[label]:
            index_cluster[index] = label

    for i,question_i in enumerate(bsl.solved_files):
        question_i = question_i[question_i.rfind('/')+1:]
        if bsl.solved_files_question_label[i].strip() == "":
            ignore_files.add(i)
            continue
        if question_i not in question_answer_encodings:
            ignore_files.add(i)
            continue
        pick_question_1 = False
        for word in bsl.solved_files_question_label[i].strip().lower().split():
            if word.isalpha():
                pick_question_1 = True
                break
        pick_question_2 = False
        for word in bsl.solved_files_question_content[i].strip().lower().split():
            if word.isalpha():
                pick_question_2 = True
                break
        if not pick_question_1 or not pick_question_2:
            ignore_files.add(i)
            out_of_language += 1
            continue

        no_english_word_title = True
        no_english_word_body = True
        
        for word in bsl.solved_files_question_label[i].strip().lower().split():
            if wordnet.synsets(word):
                no_english_word_title = False
                break
        
        for word in bsl.solved_files_question_content[i].strip().lower().split():
            if wordnet.synsets(word):
                no_english_word_body = False
                break
        
        #if (no_english_word_title or no_english_word_body) and i not in ignore_files:
        #    ignore_files.add(i)
        #    out_of_language += 1
        individual_pair_score[question_i] = {}
        if (i+1)%100 == 0:
            print "Got to index %d" %(i+1)
        for j,question_j in enumerate(bsl.solved_files):
            question_j = question_j[question_j.rfind('/')+1:]
            if bsl.solved_files_question_label[j].strip() == "":
                ignore_files.add(j)
                continue
            if question_j not in question_answer_encodings:
                ignore_files.add(j)
                continue
            if j in ignore_files:
                continue
            if i==j:
                continue
            individual_pair_score[question_i][question_j] = np.dot(question_answer_encodings[question_i],question_answer_encodings[question_j])
    #fileName = open("question_pair_scores.txt","w")
    #for question_pair in sorted(pair_score, key=pair_score.get, reverse=True):
    #    fileName.write(' '.join(question_pair).strip()+"\t"+str(pair_score[question_pair])+"\n")
    #fileName.close()
    answer_repetitions = {}
    for index,answer in enumerate(bsl.solved_files_question_answer):
        for ind,ans in enumerate(bsl.solved_files_question_answer):
            if ind <= index:
                continue
            if answer == ans and bsl.solved_files_support_docs[bsl.solved_files[index]] == bsl.solved_files_support_docs[bsl.solved_files[ind]]:
                if index not in answer_repetitions:
                    answer_repetitions[index] = [ind]
                else:
                    answer_repetitions[index].append(ind)
                if ind not in answer_repetitions:
                    answer_repetitions[ind] = [index]
                else:
                    answer_repetitions[ind].append(index)
    f_cosine = open("train_cosine.txt","w")
    f_train = open("train.txt","w")
    f_test = open("test.txt","w")
    f_test_all = open("test_all.txt","w")
    f_test_counts = open("test_counts.txt","w")
    for i,question_i in enumerate(bsl.solved_files):
        if i%250 == 0 or question_i[question_i.rfind('/')+1:] not in individual_pair_score or i in ignore_files:
            continue
        negatives = []
        for j,question_j in enumerate(bsl.solved_files):
            if i == j or question_j[question_j.rfind('/')+1:] not in individual_pair_score[question_i[question_i.rfind('/')+1:]] or j in ignore_files or index_cluster[i] == index_cluster[j]:#or (i in answer_repetitions and j in answer_repetitions[i]) or individual_pair_score[question_i[question_i.rfind('/')+1:]][question_j[question_j.rfind('/')+1:]] > 0.9:
                continue
            negatives.append(question_j[question_j.rfind('/')+1:]+"_ans")
        qid = question_i[question_i.rfind('/')+1:]
        pid = qid+"_ans"
        f_train.write(qid+"\t"+pid+"\t"+" ".join(negatives).strip()+"\n")
        f_cosine.write(pid+"\t"+pid+"\t"+pid+"\n")

    for i,question_i in enumerate(bsl.solved_files):
        if i%250!=0 or question_i[question_i.rfind('/')+1:] not in individual_pair_score or i in ignore_files:
            continue
        qid = question_i[question_i.rfind('/')+1:]
        pid = qid+"_ans"
        negatives = []
        for question_j in sorted(individual_pair_score[qid], key=individual_pair_score[qid].get, reverse=True):
            j = bsl.solved_files.index(DATA_PATH+question_j)
            if j in ignore_files:
                continue
            if qid in question_correct and question_j in question_correct[qid]:
                continue

            negatives.append(question_j+"_ans")
            if len(negatives) == 500:
                break
        pos_id = []
        if qid in question_correct:
            p_ids = list(question_correct[qid])
            for p in p_ids:
                if DATA_PATH+p not in bsl.solved_files:
                    continue
                pos_id.append(p+"_ans")
        pos_id.append(pid)
        f_test.write(qid+"\t"+' '.join(pos_id).strip()+"\t"+' '.join(negatives).strip()+"\n")
        negatives = []
        for question_j in sorted(individual_pair_score[qid], key=individual_pair_score[qid].get, reverse=True):
            negatives.append(question_j+"_ans")
            if len(negatives)% 500==0:
                f_test_all.write(qid+"\t"+pid+"\t"+' '.join(negatives).strip()+"\n")
                negatives = []
        f_test_all.write(qid+"\t"+pid+"\t"+' '.join(negatives).strip()+"\n")
        f_test_counts.write("discussions.apple.com/thread/"+qid+" has the answer with support document occurying %d times.\n" %bsl.solved_files_support_docs.values().count(bsl.solved_files_support_docs[question_i]))
    
    f_train.close()
    f_test.close()
    f_test_all.close()
    f_cosine.close()
    f_text = open("tao_model.txt","w")
    for i,solved_file in enumerate(bsl.solved_files):
        if i in ignore_files:
            continue
        question_title = bsl.solved_files_question_label[i].lower().strip()
        question_content = bsl.solved_files_question_content[i].lower().strip()
        question_id = solved_file[solved_file.rfind('/')+1:]
        if question_title == "":
            ignore_files.add(i)
            continue
        f_text.write(question_id+"\t"+question_title+"\t"+question_content+"\n")
        answer_id = question_id+"_ans"
        total_answer = bsl.solved_files_question_answer[i].lower().strip()+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[solved_file]]
        answer_title = ' '.join(total_answer.strip().split()[:100])
        answer_text = total_answer.lower().strip()
        f_text.write(answer_id+"\t"+answer_title+"\t"+answer_text+"\n")
    f_text.close()
    f_test_counts.close()
    f_ignore_files = open("ignore_files.txt","w")
    for i in ignore_files:
        f_ignore_files.write(bsl.solved_files_question_label[i]+"\t"+bsl.solved_files_question_content[i]+"\n\n")
    f_ignore_files.close()
    print "Number of out of language questions are %d" %out_of_language
    print "Number of answers repeated %d" %len(answer_repetitions)

def questions_for_lucene(bsl=None):

    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()

    lucene_file = open("questions_lucene.txt","w")
    for ind,solved_file in enumerate(bsl.solved_files):
        question_id = solved_file[solved_file.rfind('/')+1:]
        question_title = bsl.solved_files_question_label[ind]
        question_content = bsl.solved_files_question_content[ind]
        lucene_file.write(question_id+"\t"+question_title+"\t"+question_content+"\n")
    lucene_file.close()

def answers_for_lucene(bsl=None):

    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()

    lucene_file = open("answers_lucene.txt","w")
    for ind,solved_file in enumerate(bsl.solved_files):
        answer_id = solved_file[solved_file.rfind('/')+1:]+"_ans"
        answer = bsl.solved_files_question_answer[ind]+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[solved_file]]
        lucene_file.write(answer_id+"\t"+answer+"\n")
    lucene_file.close()

def write_question_answer_text(bsl=None):
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    mypath = '../DATA_qa/'
    if not os.path.exists(mypath):
        os.makedirs(mypath)
    for i,file in enumerate(bsl.solved_files):
        filename = open(mypath+file[file.rfind('/')+1:]+".txt","w")
        question = bsl.solved_files_question_text[i]
        answer = bsl.solved_files_question_answer[i]
        filename.write(question+". "+answer+"\n")
        filename.close()


def answer_encoding_label_kmeans(bsl=None):

    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()

    ans_encoding = pickle.load(open("ans_encoding.p","r"))

    X = []
    Y = []
    for ans,encoding in ans_encoding.items():
        X.append(encoding)
        Y.append(ans)
    
    topic_list = []
    topic_answers = {}
    
    for i,topic_label in enumerate(bsl.solved_files_question_topic):
        if topic_label not in topic_answers:
            topic_answers[topic_label] = [bsl.solved_files[i][bsl.solved_files[i].rfind('/')+1:]]
            topic_list.append(topic_label)
        else:
            topic_answers[topic_label].append(bsl.solved_files[i][bsl.solved_files[i].rfind('/')+1:])
    
    print "Number of topics %d" %len(topic_list)

    X_super = []
    Y_super = []

    for topic_label in topic_list:
        x_label = []
        y_label = []
        for answer in topic_answers[topic_label]:
            if answer+"_ans" in ans_encoding:
                x_label.append(ans_encoding[answer+"_ans"])
                y_label.append(answer)

        #print "Currently looking at %d answers" %len(x_label)

        labels = []
        if len(y_label) > 40:
            num_clusters = len(y_label)/40 + 1
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x_label)
            labels = kmeans.labels_
        else:
            for i in range(len(y_label)):
                labels.append(0)
        
        label_to_ind = {}
        for ind,label in enumerate(labels):
            if label not in label_to_ind:
                label_to_ind[label] = [ind]
            else:
                label_to_ind[label].append(ind)

        for label in label_to_ind:
            x_cluster = []
            y_cluster = []
            for ind in label_to_ind[label]:
                x_cluster.append(x_label[ind])
                y_cluster.append(y_label[ind])
            X_super.append(x_cluster)
            Y_super.append(y_cluster)

    print "Found %d num of clusters" %len(X_super)
    mean_super = []
    for i in range(len(X_super)):
        curr_mean = []
        for j in range(len(X_super[i])):
            if len(curr_mean) == 0:
                curr_mean = X_super[i][j]
            else:
                for k in range(len(curr_mean)):
                    curr_mean[k] += X_super[i][j][k]
        curr_mean /= len(X_super[i])
        mean_super.append(curr_mean)

    print "Found %d means" %len(mean_super)

    train_file = open("train_kmeans.txt","w")
    for i in range(len(X_super)):
        for j in range(len(X_super[i])):
            mean_scores = {}
            for ind,mean in enumerate(mean_super):
                mean_scores[ind] = np.dot(np.array(mean),np.array(X_super[i][j]))
            top_10_clusters = []
            for ind in sorted(mean_scores, key=mean_scores.get, reverse=True):
                if ind == i:
                    continue
                top_10_clusters.append(ind)
                if len(top_10_clusters) == 10:
                    break

            qid = Y_super[i][j]
            pos = [Y_super[i][j]+"_ans"]
            negs = []
            for k in top_10_clusters:
                for negative in Y_super[k]:
                    negs.append(negative+"_ans")

            train_file.write(qid+"\t"+' '.join(pos).strip()+"\t"+' '.join(negs).strip()+"\n")

    train_file.close()


def answer_encoding_magnet(bsl=None):

    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()

    num_things = 5

    considered_questions = set()

    ans_encoding = pickle.load(open("ans_encoding.p","r"))

    X = []
    Y = []
    for ans,encoding in ans_encoding.items():
        X.append(encoding)
        Y.append(ans)

    topic_list = []
    topic_answers = {}
    topic_counts = {}

    num_clusters_remerged = 0

    for i,topic_label in enumerate(bsl.solved_files_question_topic):
    #for i,solved_file in enumerate(bsl.solved_files):
    #    topic_label = bsl.solved_files_support_docs[solved_file]
        if topic_label not in topic_answers:
            topic_answers[topic_label] = [bsl.solved_files[i][bsl.solved_files[i].rfind('/')+1:]]
            topic_list.append(topic_label)
            topic_counts[topic_label] = 1
        else:
            topic_answers[topic_label].append(bsl.solved_files[i][bsl.solved_files[i].rfind('/')+1:])
            topic_counts[topic_label] += 1

    while True:
        exit_loop = False
        for topic in sorted(topic_counts, key=topic_counts.get, reverse=False):
            if topic_counts[topic] >= num_things:
                exit_loop = True
            break

        if exit_loop:
            break
        min_1 = min_2 = ""
        for topic in sorted(topic_counts, key=topic_counts.get, reverse=False):
             if min_1 == "":
                 min_1 = topic
             elif min_2 == "":
                 min_2 = topic
             else:
                 break
        topic_counts[min_2] += topic_counts[min_1]
        topic_answers[min_2] += topic_answers[min_1]
        del topic_counts[min_1]
        del topic_answers[min_1]
        topic_list.remove(min_1)

    print "Number of topics %d" %len(topic_list)

    X_super = []
    Y_super = []

    TOPIC_super = []

    for topic_label in topic_list:
        x_label = []
        y_label = []

        for answer in topic_answers[topic_label]:
            if len(topic_answers[topic_label]) < num_things:
                continue
            if answer+"_ans" in ans_encoding:
                x_label.append(ans_encoding[answer+"_ans"])
                y_label.append(answer)

        #print "Currently looking at %d answers" %len(x_label)

        labels = []
        if len(y_label) > max(40,4*num_things):
            num_clusters = len(y_label)/(max(40,4*num_things)) + 1
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x_label)
            labels = kmeans.labels_
        else:
            for i in range(len(y_label)):
                labels.append(0)

        label_to_ind = {}
        label_to_count = {}
        for ind,label in enumerate(labels):
            if label not in label_to_ind:
                label_to_ind[label] = [ind]
                label_to_count[label] = 1
            else:
                label_to_ind[label].append(ind)
                label_to_count[label] += 1

        
        for label in sorted(label_to_count, key=label_to_count.get, reverse=True):
            x_cluster = []
            y_cluster = []
            for ind in label_to_ind[label]:
                x_cluster.append(x_label[ind])
                y_cluster.append(y_label[ind])
            if len(x_cluster) < num_things:
                X_super[-1] += x_cluster
                Y_super[-1] += y_cluster
                num_clusters_remerged += 1
            else:
                X_super.append(x_cluster)
                Y_super.append(y_cluster)
                TOPIC_super.append(topic_label)

    print "Found %d num of clusters" %len(X_super)
    mean_super = []
    for i in range(len(X_super)):
        curr_mean = []
        for j in range(len(X_super[i])):
            if len(curr_mean) == 0:
                curr_mean = X_super[i][j]
            else:
                for k in range(len(curr_mean)):
                    curr_mean[k] += X_super[i][j][k]
        curr_mean /= len(X_super[i])
        mean_super.append(curr_mean)

    print "Found %d means" %len(mean_super)
    print "Number of clusters remerged are %d" %num_clusters_remerged

    num_over_shot = 0
    mean_close = {}
    for i,mean_i in enumerate(mean_super):
        mean_close[i] = []
        current_neighbours = {}
        for j,mean_j in enumerate(mean_super):
            if i==j:
                continue
            current_neighbours[j] = np.dot(np.array(mean_i),np.array(mean_j))
        topics_considered = set()
        topics_considered.add(TOPIC_super[i])
        cnt = 0
        for neighbour in sorted(current_neighbours, key=current_neighbours.get, reverse=True):
            cnt += 1
            if TOPIC_super[neighbour] in topics_considered:
                continue
            mean_close[i].append(neighbour)
            topics_considered.add(TOPIC_super[i])
            if len(mean_close[i]) == num_things-1:
                break
        if cnt>num_things-1:
            num_over_shot += 1 
    
    print "Needed to look at more clusters for %d clusters" %num_over_shot
    train_file = open("train_magnet.txt","w")
    train_imposter = open("train_imposter.txt","w")

    import copy
    Y_super_reused = [[] for _ in range(len(Y_super))]

    for I in range(1):
        for mean_ind in mean_close.keys():
            candidate_means = mean_close[mean_ind]
            sampled_guys = []
            y_pos = copy.copy(Y_super[mean_ind])
            if len(y_pos) <num_things:
                y_pos += Y_super_reused[mean_ind]
            random.shuffle(y_pos)
            assert len(y_pos) >= num_things
            y_pos = y_pos[:num_things]
            for yp in y_pos:
                if yp not in Y_super_reused[mean_ind]:
                    Y_super_reused[mean_ind].append(yp) 
                    Y_super[mean_ind].remove(yp)
            #while len(y_pos) < num_things:
            #    y_pos += y_pos
            #y_pos = y_pos[:num_things]
            sampled_guys.append(y_pos)
            for neg_ind in candidate_means:
                y_pos = copy.copy(Y_super[neg_ind])
                if len(y_pos) <num_things:
                    y_pos += Y_super_reused[neg_ind]
                random.shuffle(y_pos)
                assert len(y_pos) >= num_things
                y_pos = y_pos[:num_things]
                for yp in y_pos:
                    if yp not in Y_super_reused[neg_ind]:
                        Y_super_reused[neg_ind].append(yp)
                        Y_super[neg_ind].remove(yp)
                sampled_guys.append(y_pos)
            for i in range(len(sampled_guys)):
                negatives = []
                for j in range(len(sampled_guys)):
                    if i==j:
                        continue
                    negatives = negatives + sampled_guys[j]
                positive = copy.copy(sampled_guys[i])
                for j in range(len(negatives)):
                    negatives[j] += "_ans"
                for j in range(len(positive)):
                    positive[j] += "_ans"
                for j in range(len(positive)):
                    curr_pos = copy.copy(positive)
                    del curr_pos[j]
                    curr_pos = [ positive[j] ] + curr_pos
                    train_file.write(positive[j][:positive[j].find('_')]+"\t"+' '.join(curr_pos).strip()+"\t"+' '.join(negatives).strip()+"\n")
                    train_imposter.write(positive[j][:positive[j].find('_')]+"\t"+positive[j]+"\t"+' '.join(negatives).strip()+"\n")
                    considered_questions.add(positive[j][:positive[j].find('_')])

    train_file.close()
    train_imposter.close()
    print "Number of questions considered are %d" %len(considered_questions)

def answer_encoding_kmeans(bsl=None):

    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()

    ans_encoding = pickle.load(open("ans_encoding.p","r"))

    X = []
    Y = []
    for ans,encoding in ans_encoding.items():
        X.append(encoding)
        Y.append(ans)
   
    print "The number of answers: %d" %len(Y)
    kmeans = KMeans(n_clusters=len(Y)/100, random_state=0).fit(X)
    print "Completed KMeans"
    centers = kmeans.cluster_centers_
    center_neighbours = {}
    for i,center_i in enumerate(centers):
        neighbours_scores = {}
        for j,center_j in enumerate(centers):
            if i == j:
                continue
            neighbours_scores[tuple(center_j)] = np.dot(center_i,center_j)
        center_neighbours[tuple(center_i)] = []
        for neighbour in sorted(neighbours_scores, key=neighbours_scores.get, reverse = True):
            center_neighbours[tuple(center_i)].append(neighbour)
            if len(center_neighbours[tuple(center_i)]) == 10:
                break
    ans_to_center = {}
    center_to_ans = {}

    for ans,encoding in ans_encoding.items():
        ans_to_center[ans] = centers[0]
        for center in centers:
            if np.dot(center,encoding) > np.dot(ans_to_center[ans],encoding):
                ans_to_center[ans] = center
        if tuple(ans_to_center[ans]) not in center_to_ans:
            center_to_ans[tuple(ans_to_center[ans])] = [ans]
        else:
            center_to_ans[tuple(ans_to_center[ans])].append(ans)
    question_correct = pickle.load(open("question_correct.p","r"))
    f_train = open("train.txt","w")

    for I in range(1):
        for i,ans_i in enumerate(Y):
            negatives = []
            qid = ans_i[:ans_i.rfind('_')]
            center_i = ans_to_center[ans_i]
            #for j,ans_j in enumerate(Y):
                #if kmeans.labels_[i] != kmeans.labels_[j] and not (qid in question_correct and ans_j[:ans_j.rfind('_')] in question_correct[qid]):
                #center_j = ans_to_center[ans_j]
                #if tuple(center_j) in center_neighbours[tuple(center_i)]:
                #    negatives.append(ans_j)
            consider = []
            for center_j in center_neighbours[tuple(center_i)]:
                if center_j not in center_to_ans:
                    #print "Haila, this center has no friends"
                    continue
                consider = consider + center_to_ans[center_j]
            for j,ans_j in enumerate(consider):
                if not (qid in question_correct and ans_j[:ans_j.rfind('_')] in question_correct[qid]):
                    negatives.append(ans_j)
            positives = []
            if qid in question_correct:
                for p in question_correct[qid]:
                    positives.append(p+"_ans")
            positives.append(ans_i)
            f_train.write(qid+"\t"+' '.join(positives).strip()+"\t"+' '.join(negatives).strip()+"\n")
    f_train.close()
    print "Number of centers are %d, centers with neighbours %d" %(len(centers),len(center_to_ans))

def random_50_pairs(bsl=None):
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()

    import copy

    answer_type_1 = copy.copy(bsl.solved_files)
    random.shuffle(answer_type_1)
    f = open("compare_answers.txt","w")
    for i in range(10):
        for j in range(10):
            if j<=i:
                continue
            f.write(str(i*10+j)+"\n"+bsl.solved_files_question_answer[bsl.solved_files.index(answer_type_1[i])]+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[answer_type_1[i]]]+"\n"+bsl.solved_files_question_answer[bsl.solved_files.index(answer_type_1[j])]+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[answer_type_1[j]]]+"\n\n")
    f.close()

def question_topic_counts(bsl=None):

    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()

    question_topic_counts = {}
    for topic in bsl.solved_files_question_topic:
        if topic not in question_topic_counts:
            question_topic_counts[topic] = 1
        else:
            question_topic_counts[topic] += 1
    
    topic_count = open("topic_counts.txt","w")
    for topic in sorted(question_topic_counts, key = question_topic_counts.get, reverse=True):
        topic_count.write(topic+"\t"+str(question_topic_counts[topic])+"\n")
    topic_count.close()

def tf_idf_results(bsl=None):

    question_document_frequency = {}
    question_word_frequency = []
    answer_document_frequency = {}
    answer_word_frequency = []
    word_set = set()

    for i in range(len(bsl.solved_files)):
        if i%100 == 0:
            print "Got to index %d" %(i+1)
        question = bsl.solved_files_question_text[i].lower()
        answer = bsl.solved_files_question_answer[i].lower()+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[bsl.solved_files[i]]].lower()
        question_words = question.split()
        answer_words = answer.split()
        question_words_counts = {}
        for q_w in question_words:
            word_set.add(q_w)
            if q_w not in question_words_counts:
                if q_w not in question_document_frequency:
                    question_document_frequency[q_w] = 1
                else:
                    question_document_frequency[q_w] += 1
                question_words_counts[q_w] = 1
            else:
                question_words_counts[q_w] += 1 
        answer_words_counts = {}
        for a_w in answer_words:
            word_set.add(a_w)
            if a_w not in answer_words_counts:
                if a_w not in answer_document_frequency:
                    answer_document_frequency[a_w] = 1
                else:
                    answer_document_frequency[a_w] += 1
                answer_words_counts[a_w] = 1
            else:
                answer_words_counts[a_w] += 1
        question_word_frequency.append(question_words_counts)  
        answer_word_frequency.append(answer_words_counts)

    print "A total of %d words considered" %len(word_set)

    word_list = list(word_set)

    answer_vectors = []

    for i in range(len(answer_word_frequency)):
        vector = [0.0 for _ in range(len(word_list))]
        for word in answer_word_frequency[i]:
            doc_count = answer_document_frequency[word]
            if word in question_document_frequency:
                doc_count += question_document_frequency[word]
            vector[word_list.index(word)] = float(answer_word_frequency[i][word])*math.log(2*len(answer_word_frequency)/float(doc_count))
        answer_vectors.append(np.array(vector))
        if (i+1)%100 == 0:
            print "Created for answer number %d" %(i+1)

    question_vectors = []

    for i in range(len(question_word_frequency)):
        vector = [0.0 for _ in range(len(word_list))]
        for word in question_word_frequency[i]:
            doc_count = question_document_frequency[word]
            if word in answer_document_frequency:
                doc_count += answer_document_frequency[word]
            vector[word_list.index(word)] = float(question_word_frequency[i][word])*math.log(2*len(question_word_frequency)/float(doc_count))
        question_vectors.append(np.array(vector))
        if (i+1)%100 == 0:
            print "Created for question number %d" %(i+1)

    file_test = open("test.txt","r")
    lines = file_test.readlines()
    test_questions = []
    test_answers = []
    test_candidates = []
    for line in lines:
        line = line.strip()
        line = line.split('\t')
        test_questions.append(bsl.solved_files.index(DATA_PATH+line[0]))
        pos_candidates = line[1].split()
        neg_candidates = line[2].split()
        for i in range(len(pos_candidates)):
            pos_candidates[i] = pos_candidates[i][:pos_candidates[i].rfind('_')]
            pos_candidates[i] = bsl.solved_files.index(DATA_PATH+pos_candidates[i])
        for i in range(len(neg_candidates)):
            neg_candidates[i] = neg_candidates[i][:neg_candidates[i].rfind('_')]
            neg_candidates[i] = bsl.solved_files.index(DATA_PATH+neg_candidates[i])
        test_answers.append(set(pos_candidates))
        test_candidates.append(pos_candidates+neg_candidates)

    number_correct = 0

    for i in range(len(test_questions)):
        question_vector = question_vectors[test_questions[i]]
        max_cosine = 0.0
        max_index = 0.0
        for j in range(len(test_candidates[i])):
            answer_vector = answer_vectors[test_candidates[i][j]]
            cosine = np.dot(question_vector,answer_vector)/(np.linalg.norm(answer_vector)*np.linalg.norm(question_vector))
            if cosine>max_cosine:
                max_cosine = cosine
                max_index = test_candidates[i][j]
        if max_index in test_answers[i]:
            number_correct += 1

    print "Number correct %d" %number_correct 

def tf_idf_question_answer(bsl=None):
    
    question_document_frequency = {}
    question_word_frequency = []
    answer_document_frequency = {}
    answer_word_frequency = []
    word_set = set()

    for i in range(len(bsl.solved_files)):
        if i%100 == 0:
            print "Got to index %d" %(i+1)
        question = bsl.solved_files_question_text[i].lower()
        answer = bsl.solved_files_question_answer[i].lower()+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[bsl.solved_files[i]]].lower()
        question_words = question.split()
        answer_words = answer.split()
        question_words_counts = {}
        for q_w in question_words:
            word_set.add(q_w)
            if q_w not in question_words_counts:
                if q_w not in question_document_frequency:
                    question_document_frequency[q_w] = 1
                else:
                    question_document_frequency[q_w] += 1
                question_words_counts[q_w] = 1
            else:
                question_words_counts[q_w] += 1 
        answer_words_counts = {}
        for a_w in answer_words:
            word_set.add(a_w)
            if a_w not in answer_words_counts:
                if a_w not in answer_document_frequency:
                    answer_document_frequency[a_w] = 1
                else:
                    answer_document_frequency[a_w] += 1
                answer_words_counts[a_w] = 1
            else:
                answer_words_counts[a_w] += 1
        question_word_frequency.append(question_words_counts)  
        answer_word_frequency.append(answer_words_counts)

    print "A total of %d words considered" %len(word_set)

    word_list = list(word_set)

    answer_vectors = []

    for i in range(len(answer_word_frequency)):
        vector = [0.0 for _ in range(len(word_list))]
        for word in answer_word_frequency[i]:
            vector[word_list.index(word)] = float(answer_word_frequency[i][word])*math.log(len(answer_word_frequency)/float(answer_document_frequency[word]))
        answer_vectors.append(np.array(vector))
        if (i+1)%100 == 0:
            print "Created for answer number %d" %(i+1)
    
     #indices = [i for i in range(len(answer_vectors))]
     #np.shuffle(indices)
    pair_scores = {}
    pair_scores_list = {}
    for i in range(len(answer_vectors)):
        if i%100 == 0:
            print "Got to index %d" %(i+1)
        for j in range(len(answer_vectors)):
            if j<=i:
                continue
            pair_scores[tuple([i,j])] = np.dot(answer_vectors[i],answer_vectors[j])/(np.linalg.norm(answer_vectors[i])*np.linalg.norm(answer_vectors[j]))
            if i not in pair_scores_list:
                pair_scores_list[i] = [pair_scores[tuple([i,j])]]
            else:
                pair_scores_list[i].append(pair_scores[tuple([i,j])])
    top_2000_pairs = open("similar_answers.txt","w")
    cnt = 0
    print "Beginning the task of getting the top 20000 pairs"

    answer_file = open("answer_tf_idf.p","wb")
    pickle.dump(pair_scores_list,answer_file)
    answer_file.close()

    for pair in sorted(pair_scores, key=pair_scores.get, reverse=True):
        cnt += 1
        i = pair[0]
        j = pair[1]
        if cnt%200 == 0:
            print "Wrote %d similar pairs" %cnt
        top_2000_pairs.write(str(cnt)+"\n"+bsl.solved_files_question_answer[i].lower()+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[bsl.solved_files[i]]].lower()+"\n"+bsl.solved_files_question_answer[j].lower()+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[bsl.solved_files[j]]].lower()+"\t"+str(pair_scores[pair])+"\n\n")
        if cnt == 200000 or pair_scores[pair] < 0.0:
            break
    top_2000_pairs.close() 


def tf_idf_kmeans_answer(bsl=None):
    
    question_document_frequency = {}
    question_word_frequency = []
    answer_document_frequency = {}
    answer_word_frequency = []
    word_set = set()

    word_counts = {}

    for i in range(len(bsl.solved_files)):
        question = bsl.solved_files_question_text[i].lower()
        answer = bsl.solved_files_question_answer[i].lower()+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[bsl.solved_files[i]]].lower()
        question_words = question.split()
        answer_words = answer.split()
        for q_w in question_words:
            if q_w not in word_counts:
                word_counts[q_w] = 1
            else:
                word_counts[q_w] += 1
        for a_w in answer_words:
            if a_w not in word_counts:
                word_counts[a_w] = 1
            else:
                word_counts[a_w] += 1 

    for i in range(len(bsl.solved_files)):
        if i%100 == 0:
            print "Got to index %d" %(i+1)
        question = bsl.solved_files_question_text[i].lower()
        answer = bsl.solved_files_question_answer[i].lower()+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[bsl.solved_files[i]]].lower()
        question_words = question.split()
        answer_words = answer.split()
        question_words_counts = {}
        for q_w in question_words:
            if word_counts[q_w] < 5:
                continue
            word_set.add(q_w)
            if q_w not in question_words_counts:
                if q_w not in question_document_frequency:
                    question_document_frequency[q_w] = 1
                else:
                    question_document_frequency[q_w] += 1
                question_words_counts[q_w] = 1
            else:
                question_words_counts[q_w] += 1 
        answer_words_counts = {}
        for a_w in answer_words:
            if word_counts[a_w] < 5:
                continue
            word_set.add(a_w)
            if a_w not in answer_words_counts:
                if a_w not in answer_document_frequency:
                    answer_document_frequency[a_w] = 1
                else:
                    answer_document_frequency[a_w] += 1
                answer_words_counts[a_w] = 1
            else:
                answer_words_counts[a_w] += 1
        question_word_frequency.append(question_words_counts)  
        answer_word_frequency.append(answer_words_counts)

    print "A total of %d words considered" %len(word_set)

    word_list = list(word_set)

    answer_vectors = []

    for i in range(len(answer_word_frequency)):
        vector = [0.0 for _ in range(len(word_list))]
        for word in answer_word_frequency[i]:
            vector[word_list.index(word)] = float(answer_word_frequency[i][word])*math.log(len(answer_word_frequency)/float(answer_document_frequency[word]))
        answer_vectors.append(np.array(vector))
        if (i+1)%100 == 0:
            print "Created for answer number %d" %(i+1)
    kmeans = KMeans(n_clusters=1000, random_state=0).fit(answer_vectors)
    labels = kmeans.labels_
    clusters = {}
    cluster_sizes = {}
    for ind,label in enumerate(labels):
        if label not in clusters:
            clusters[label] = [ind]
            cluster_sizes[label] = 1
        else:
            clusters[label].append(ind)
            cluster_sizes[label] += 1

    print "Total number of clusters found are %d" %len(clusters)
    file_answers = open("tf_idf_kmeans_answers.txt","w")
    cnt = 0
    for label in sorted(cluster_sizes, key=cluster_sizes.get, reverse=True):
         cnt += 1
         file_answers.write("Cluster size %d. Cluster number %d.\n"%(cluster_sizes[label],cnt))
         for ans_ind in clusters[label]:
             file_answers.write("Answer:\n%s %s\n\n"%(bsl.solved_files_question_answer[ans_ind],bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[bsl.solved_files[ans_ind]]]))
    file_answers.close()
    pickle.dump(answer_clusters,open("answer_clusters.p","wb"))

def tf_idf_kmeans_question(bsl=None):
    
    question_document_frequency = {}
    question_word_frequency = []
    answer_document_frequency = {}
    answer_word_frequency = []
    word_set = set()

    word_counts = {}

    for i in range(len(bsl.solved_files)):
        question = bsl.solved_files_question_text[i].lower()
        answer = bsl.solved_files_question_answer[i].lower()+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[bsl.solved_files[i]]].lower()
        question_words = question.split()
        answer_words = answer.split()
        for q_w in question_words:
            if q_w not in word_counts:
                word_counts[q_w] = 1
            else:
                word_counts[q_w] += 1
        for a_w in answer_words:
            if a_w not in word_counts:
                word_counts[a_w] = 1
            else:
                word_counts[a_w] += 1 

    for i in range(len(bsl.solved_files)):
        if i%100 == 0:
            print "Got to index %d" %(i+1)
        question = bsl.solved_files_question_text[i].lower()
        answer = bsl.solved_files_question_answer[i].lower()+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[bsl.solved_files[i]]].lower()
        question_words = question.split()
        answer_words = answer.split()
        question_words_counts = {}
        for q_w in question_words:
            if word_counts[q_w] < 5:
                continue
            word_set.add(q_w)
            if q_w not in question_words_counts:
                if q_w not in question_document_frequency:
                    question_document_frequency[q_w] = 1
                else:
                    question_document_frequency[q_w] += 1
                question_words_counts[q_w] = 1
            else:
                question_words_counts[q_w] += 1 
        answer_words_counts = {}
        for a_w in answer_words:
            if word_counts[a_w] < 5:
                continue
            word_set.add(a_w)
            if a_w not in answer_words_counts:
                if a_w not in answer_document_frequency:
                    answer_document_frequency[a_w] = 1
                else:
                    answer_document_frequency[a_w] += 1
                answer_words_counts[a_w] = 1
            else:
                answer_words_counts[a_w] += 1
        question_word_frequency.append(question_words_counts)  
        answer_word_frequency.append(answer_words_counts)

    print "A total of %d words considered" %len(word_set)

    word_list = list(word_set)

    question_vectors = []

    for i in range(len(question_word_frequency)):
        vector = [0.0 for _ in range(len(word_list))]
        for word in question_word_frequency[i]:
            vector[word_list.index(word)] = float(question_word_frequency[i][word])*math.log(len(question_word_frequency)/float(question_document_frequency[word]))
        question_vectors.append(np.array(vector))
        if (i+1)%100 == 0:
            print "Created for answer number %d" %(i+1)
    kmeans = KMeans(n_clusters=1000, random_state=0).fit(question_vectors)
    labels = kmeans.labels_
    clusters = {}
    cluster_sizes = {}
    for ind,label in enumerate(labels):
        if label not in clusters:
            clusters[label] = [ind]
            cluster_sizes[label] = 1
        else:
            clusters[label].append(ind)
            cluster_sizes[label] += 1

    print "Total number of clusters found are %d" %len(clusters)
    file_answers = open("tf_idf_kmeans_questions.txt","w")
    cnt = 0
    for label in sorted(cluster_sizes, key=cluster_sizes.get, reverse=True):
         cnt += 1
         file_answers.write("Cluster size %d. Cluster number %d.\n"%(cluster_sizes[label],cnt))
         for ans_ind in clusters[label]:
             file_answers.write("Question:\n%s %s\n\n"%(bsl.solved_files_question_text[ans_ind],bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[bsl.solved_files[ans_ind]]]))
    file_answers.close()
    pickle.dump(clusters,open("question_clusters.p","wb"))

def tf_idf_question_question(bsl=None):
    
    question_document_frequency = {}
    question_word_frequency = []
    answer_document_frequency = {}
    answer_word_frequency = []
    word_set = set()

    for i in range(len(bsl.solved_files)):
        if i%100 == 0:
            print "Got to index %d" %(i+1)
        question = bsl.solved_files_question_text[i].lower()
        answer = bsl.solved_files_question_answer[i].lower()+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[bsl.solved_files[i]]].lower()
        question_words = question.split()
        answer_words = answer.split()
        question_words_counts = {}
        for q_w in question_words:
            word_set.add(q_w)
            if q_w not in question_words_counts:
                if q_w not in question_document_frequency:
                    question_document_frequency[q_w] = 1
                else:
                    question_document_frequency[q_w] += 1
                question_words_counts[q_w] = 1
            else:
                question_words_counts[q_w] += 1 
        answer_words_counts = {}
        for a_w in answer_words:
            word_set.add(a_w)
            if a_w not in answer_words_counts:
                if a_w not in answer_document_frequency:
                    answer_document_frequency[a_w] = 1
                else:
                    answer_document_frequency[a_w] += 1
                answer_words_counts[a_w] = 1
            else:
                answer_words_counts[a_w] += 1
        question_word_frequency.append(question_words_counts)  
        answer_word_frequency.append(answer_words_counts)

    print "A total of %d words considered" %len(word_set)

    print "Total number of questions %d" %len(question_word_frequency)

    word_list = list(word_set)

    question_vectors = []

    for i in range(len(question_word_frequency)):
        vector = [0.0 for _ in range(len(word_list))]
        for word in question_word_frequency[i]:
            vector[word_list.index(word)] = float(question_word_frequency[i][word])*math.log(len(question_word_frequency)/float(question_document_frequency[word]))
        question_vectors.append(np.array(vector))
        if (i+1)%100 == 0:
            print "Created for question number %d" %(i+1)
    
     #indices = [i for i in range(len(answer_vectors))]
     #np.shuffle(indices)
    pair_scores = {}
    for i in range(len(question_vectors)):
        if i%100 == 0:
            print "Got to index %d" %(i+1)
        for j in range(len(question_vectors)):
            if j<=i:
                continue
            pair_scores[tuple([i,j])] = np.dot(question_vectors[i],question_vectors[j])#/(np.linalg.norm(question_vectors[i])*np.linalg.norm(question_vectors[j]))
            if pair_scores[tuple([i,j])] != 0.0:
                pair_scores[tuple([i,j])] /= (np.linalg.norm(question_vectors[i])*np.linalg.norm(question_vectors[j]))
            if bsl.solved_files_question_text[i] == bsl.solved_files_question_text[j]:
                print "Repeated question found"
                print pair_scores[tuple([i,j])]
                print bsl.solved_files_question_text[i],bsl.solved_files_question_text[j]
    top_2000_pairs = open("similar_questions.txt","w")
    cnt = 0
    print "Beginning the task of getting the top 20000 pairs"
    for pair in sorted(pair_scores, key=pair_scores.get, reverse=True):
        cnt += 1
        i = pair[0]
        j = pair[1]
        if cnt%200 == 0:
            print "Wrote %d similar pairs" %cnt
        top_2000_pairs.write(str(cnt)+"\n"+bsl.solved_files_question_label[i].lower()+" "+bsl.solved_files_question_content[i]+"\n"+bsl.solved_files_question_label[j].lower()+" "+bsl.solved_files_question_content[j].lower()+"\t"+str(pair_scores[pair])+"\n\n")
        if cnt == 200000 or pair_scores[pair] < 0.0:
            break
    top_2000_pairs.close()
    question_tf_idf = open("question_tf_idf.p","wb")
    pickle.dump(pair_scores,question_tf_idf)
    question_tf_idf.close()

def sample_tf_idf_answer(bsl=None):
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    indices = [i for i in range(len(bsl.solved_files))]
    random.shuffle(indices)
    indices = indices[:100]

    question_document_frequency = {}
    question_word_frequency = []
    answer_document_frequency = {}
    answer_word_frequency = []
    word_set = set()

    for i in range(len(bsl.solved_files)):
        if i%100 == 0:
            print "Got to index %d" %(i+1)
        question = bsl.solved_files_question_text[i].lower()
        answer = bsl.solved_files_question_answer[i].lower()+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[bsl.solved_files[i]]].lower()
        question_words = question.split()
        answer_words = answer.split()
        question_words_counts = {}
        for q_w in question_words:
            word_set.add(q_w)
            if q_w not in question_words_counts:
                if q_w not in question_document_frequency:
                    question_document_frequency[q_w] = 1
                else:
                    question_document_frequency[q_w] += 1
                question_words_counts[q_w] = 1
            else:
                question_words_counts[q_w] += 1
        answer_words_counts = {}
        for a_w in answer_words:
            word_set.add(a_w)
            if a_w not in answer_words_counts:
                if a_w not in answer_document_frequency:
                    answer_document_frequency[a_w] = 1
                else:
                    answer_document_frequency[a_w] += 1
                answer_words_counts[a_w] = 1
            else:
                answer_words_counts[a_w] += 1
        question_word_frequency.append(question_words_counts)
        answer_word_frequency.append(answer_words_counts)

    print "A total of %d words considered" %len(word_set)

    word_list = list(word_set)

    answer_vectors = []

    for i in range(len(answer_word_frequency)):
        vector = [0.0 for _ in range(len(word_list))]
        for word in answer_word_frequency[i]:
            vector[word_list.index(word)] = float(answer_word_frequency[i][word])*math.log(len(answer_word_frequency)/float(answer_document_frequency[word]))
        answer_vectors.append(np.array(vector))
        if (i+1)%100 == 0:
            print "Created for answer number %d" %(i+1)

     #indices = [i for i in range(len(answer_vectors))]
     #np.shuffle(indices)
    pair_scores = {}
    for i in range(len(answer_vectors)):
        if i%100 == 0:
            print "Got to index %d" %(i+1)
        for j in range(len(answer_vectors)):
            if j<=i:
                continue
            pair_scores[tuple([i,j])] = np.dot(answer_vectors[i],answer_vectors[j])/(np.linalg.norm(answer_vectors[i])*np.linalg.norm(answer_vectors[j]))
    answer_tf_idf = pair_scores
    pair_scores_list = []

    for index in indices:
        pair_scores = {}
        for neighbour in range(len(bsl.solved_files)):
            if neighbour == index:
                continue
            key = []
            if neighbour < index:
                key = tuple([neighbour,index])
            else:
                key = tuple([index,neighbour])
            pair_scores[neighbour] = answer_tf_idf[key]
        pair_scores_list.append(pair_scores)
    print "Collected neighbours for 100 random answers"
    
    f_answers = open("100_close_answers.txt","w")
    for ind,index in enumerate(indices):
        pair_scores = pair_scores_list[ind]
        f_answers.write("Random answer number " +str(ind+1)+"\n."+bsl.solved_files_question_answer[index]+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[bsl.solved_files[index]]]+"\n")
        cnt = 0
        for neighbour in sorted(pair_scores, key=pair_scores.get, reverse=True):
            f_answers.write(bsl.solved_files_question_answer[neighbour]+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[bsl.solved_files[neighbour]]]+" "+str(pair_scores[neighbour])+"\n")
            cnt += 1
            if cnt == 20:
                break
    f_answers.close()

def sample_tf_idf_question(bsl=None):
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()
    indices = [i for i in range(len(bsl.solved_files))]
    random.shuffle(indices)
    indices = indices[:100]

    question_document_frequency = {}
    question_word_frequency = []
    answer_document_frequency = {}
    answer_word_frequency = []
    word_set = set()

    for i in range(len(bsl.solved_files)):
        if i%100 == 0:
            print "Got to index %d" %(i+1)
        question = bsl.solved_files_question_text[i].lower()
        answer = bsl.solved_files_question_answer[i].lower()+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[bsl.solved_files[i]]].lower()
        question_words = question.split()
        answer_words = answer.split()
        question_words_counts = {}
        for q_w in question_words:
            word_set.add(q_w)
            if q_w not in question_words_counts:
                if q_w not in question_document_frequency:
                    question_document_frequency[q_w] = 1
                else:
                    question_document_frequency[q_w] += 1
                question_words_counts[q_w] = 1
            else:
                question_words_counts[q_w] += 1
        answer_words_counts = {}
        for a_w in answer_words:
            word_set.add(a_w)
            if a_w not in answer_words_counts:
                if a_w not in answer_document_frequency:
                    answer_document_frequency[a_w] = 1
                else:
                    answer_document_frequency[a_w] += 1
                answer_words_counts[a_w] = 1
            else:
                answer_words_counts[a_w] += 1
        question_word_frequency.append(question_words_counts)
        answer_word_frequency.append(answer_words_counts)

    print "A total of %d words considered" %len(word_set)

    word_list = list(word_set)

    question_vectors = []

    for i in range(len(question_word_frequency)):
        vector = [0.0 for _ in range(len(word_list))]
        for word in question_word_frequency[i]:
            vector[word_list.index(word)] = float(question_word_frequency[i][word])*math.log(len(question_word_frequency)/float(question_document_frequency[word]))
        question_vectors.append(np.array(vector))
        if (i+1)%100 == 0:
            print "Created for answer number %d" %(i+1)

     #indices = [i for i in range(len(answer_vectors))]
     #np.shuffle(indices)
    pair_scores = {}
    for i in range(len(question_vectors)):
        if i%100 == 0:
            print "Got to index %d" %(i+1)
        for j in range(len(question_vectors)):
            if j<=i:
                continue
            pair_scores[tuple([i,j])] = np.dot(question_vectors[i],question_vectors[j])
            if pair_scores[tuple([i,j])] != 0.0:
            	pair_scores[tuple([i,j])] /= (np.linalg.norm(question_vectors[i])*np.linalg.norm(question_vectors[j]))
    question_tf_idf = pair_scores
    pair_scores_list = []

    for index in indices:
        pair_scores = {}
        for neighbour in range(len(bsl.solved_files)):
            if neighbour == index:
                continue
            key = []
            if neighbour < index:
                key = tuple([neighbour,index])
            else:
                key = tuple([index,neighbour])
            pair_scores[neighbour] = question_tf_idf[key]
        pair_scores_list.append(pair_scores)
    print "Collected neighbours for 100 random questions"
    
    f_question = open("100_close_questions.txt","w")
    for ind,index in enumerate(indices):
        pair_scores = pair_scores_list[ind]
        f_question.write("Random question number " +str(ind+1)+"\n."+' '.join(bsl.solved_files_question_text[index].split()[:500]).strip()+"\n")
        cnt = 0
        for neighbour in sorted(pair_scores, key=pair_scores.get, reverse=True):
            f_question.write(' '.join(bsl.solved_files_question_text[neighbour].split()[:500]).strip()+" "+str(pair_scores[neighbour])+"\n")
            cnt += 1
            if cnt == 20:
                break
    f_question.close()

def similar_questions_different_answers(bsl=None):
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()

    question_document_frequency = {}
    question_word_frequency = []
    answer_document_frequency = {}
    answer_word_frequency = []
    word_set = set()

    for i in range(len(bsl.solved_files)):
        if i%100 == 0:
            print "Got to index %d" %(i+1)
        question = bsl.solved_files_question_text[i].lower()
        answer = bsl.solved_files_question_answer[i].lower()+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[bsl.solved_files[i]]].lower()
        question_words = question.split()
        answer_words = answer.split()
        question_words_counts = {}
        for q_w in question_words:
            word_set.add(q_w)
            if q_w not in question_words_counts:
                if q_w not in question_document_frequency:
                    question_document_frequency[q_w] = 1
                else:
                    question_document_frequency[q_w] += 1
                question_words_counts[q_w] = 1
            else:
                question_words_counts[q_w] += 1
        answer_words_counts = {}
        for a_w in answer_words:
            word_set.add(a_w)
            if a_w not in answer_words_counts:
                if a_w not in answer_document_frequency:
                    answer_document_frequency[a_w] = 1
                else:
                    answer_document_frequency[a_w] += 1
                answer_words_counts[a_w] = 1
            else:
                answer_words_counts[a_w] += 1
        question_word_frequency.append(question_words_counts)
        answer_word_frequency.append(answer_words_counts)

    print "A total of %d words considered" %len(word_set)

    word_list = list(word_set)

    question_vectors = []

    for i in range(len(question_word_frequency)):
        vector = [0.0 for _ in range(len(word_list))]
        for word in question_word_frequency[i]:
            vector[word_list.index(word)] = float(question_word_frequency[i][word])*math.log(len(question_word_frequency)/float(question_document_frequency[word]))
        question_vectors.append(np.array(vector))
        if (i+1)%100 == 0:
            print "Created for answer number %d" %(i+1)

     #indices = [i for i in range(len(answer_vectors))]
     #np.shuffle(indices)
    pair_scores = {}
    similar_question_pairs = {}
    for i in range(len(question_vectors)):
        if i%100 == 0:
            print "Got to index %d" %(i+1)
        for j in range(len(question_vectors)):
            if j<=i:
                continue
            pair_scores[tuple([i,j])] = np.dot(question_vectors[i],question_vectors[j])
            if pair_scores[tuple([i,j])] != 0.0:
                pair_scores[tuple([i,j])] /= (np.linalg.norm(question_vectors[i])*np.linalg.norm(question_vectors[j]))
            if pair_scores[tuple([i,j])] >0.8:
                similar_question_pairs[tuple([i,j])] = pair_scores[tuple([i,j])]
    similar_questions = pickle.load(open("question_clusters.p","rb"))
    similar_answers = pickle.load(open("answer_clusters.p","rb"))
    index_answer_clusters = {}
    for cluster in similar_answers:
        for index in similar_answers[cluster]:
            index_answer_clusters[index] = cluster
    different_answer_clusters = 0
    different_pairs = open("different_pairs.txt","w")
    for ind_i,ind_j in similar_question_pairs:
        if index_answer_clusters[ind_i] != index_answer_clusters[ind_j]:
            different_answer_clusters += 1
            if different_answer_clusters%100 == 0:
                print "Found %d different answer, similar question pairs" %different_answer_clusters
            different_pairs.write("Pair "+str(different_answer_clusters)+"\n"+' '.join(bsl.solved_files_question_text[ind_i].split()[:500])+"\n"+' '.join(bsl.solved_files_question_answer[ind_i].split()[:500])+"\n"+' '.join(bsl.solved_files_question_text[ind_j].split()[:500])+"\n"+' '.join(bsl.solved_files_question_answer[ind_j].split()[:500])+"\n\n")
    different_pairs.close()

def similar_answers_different_questions(bsl=None):
    
    if not bsl:
        bsl = Baseline(sys.maxint)
        bsl.populate_data()

    question_document_frequency = {}
    question_word_frequency = []
    answer_document_frequency = {}
    answer_word_frequency = []
    word_set = set()

    for i in range(len(bsl.solved_files)):
        if i%100 == 0:
            print "Got to index %d" %(i+1)
        question = bsl.solved_files_question_text[i].lower()
        answer = bsl.solved_files_question_answer[i].lower()+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[bsl.solved_files[i]]].lower()
        question_words = question.split()
        answer_words = answer.split()
        question_words_counts = {}
        for q_w in question_words:
            word_set.add(q_w)
            if q_w not in question_words_counts:
                if q_w not in question_document_frequency:
                    question_document_frequency[q_w] = 1
                else:
                    question_document_frequency[q_w] += 1
                question_words_counts[q_w] = 1
            else:
                question_words_counts[q_w] += 1
        answer_words_counts = {}
        for a_w in answer_words:
            word_set.add(a_w)
            if a_w not in answer_words_counts:
                if a_w not in answer_document_frequency:
                    answer_document_frequency[a_w] = 1
                else:
                    answer_document_frequency[a_w] += 1
                answer_words_counts[a_w] = 1
            else:
                answer_words_counts[a_w] += 1
        question_word_frequency.append(question_words_counts)
        answer_word_frequency.append(answer_words_counts)

    print "A total of %d words considered" %len(word_set)

    word_list = list(word_set)

    answer_vectors = []

    for i in range(len(answer_word_frequency)):
        vector = [0.0 for _ in range(len(word_list))]
        for word in answer_word_frequency[i]:
            vector[word_list.index(word)] = float(answer_word_frequency[i][word])*math.log(len(answer_word_frequency)/float(answer_document_frequency[word]))
        answer_vectors.append(np.array(vector))
        if (i+1)%100 == 0:
            print "Created for answer number %d" %(i+1)

     #indices = [i for i in range(len(answer_vectors))]
     #np.shuffle(indices)
    pair_scores = {}
    similar_answer_pairs = {}
    for i in range(len(answer_vectors)):
        if i%100 == 0:
            print "Got to index %d" %(i+1)
        for j in range(len(answer_vectors)):
            if j<=i:
                continue
            pair_scores[tuple([i,j])] = np.dot(answer_vectors[i],answer_vectors[j])
            if pair_scores[tuple([i,j])] != 0.0:
                pair_scores[tuple([i,j])] /= (np.linalg.norm(answer_vectors[i])*np.linalg.norm(answer_vectors[j]))
            if pair_scores[tuple([i,j])] >0.9:
                similar_answer_pairs[tuple([i,j])] = pair_scores[tuple([i,j])]

    similar_questions = pickle.load(open("question_clusters.p","rb"))
    similar_answers = pickle.load(open("answer_clusters.p","rb"))
    index_question_clusters = {}
    for cluster in similar_questions:
        for index in similar_questions[cluster]:
            index_question_clusters[index] = cluster
    different_answer_clusters = 0
    different_pairs = open("different_answer_pairs.txt","w")
    for ind_i,ind_j in similar_answer_pairs:
        if index_question_clusters[ind_i] != index_question_clusters[ind_j]:
            different_answer_clusters += 1
            if different_answer_clusters%100 == 0:
                print "Found %d different answer, similar question pairs" %different_answer_clusters
            different_pairs.write("Pair "+str(different_answer_clusters)+"\n"+' '.join(bsl.solved_files_question_text[ind_i].split()[:500])+"\n"+' '.join(bsl.solved_files_question_answer[ind_i].split()[:500])+"\n"+' '.join(bsl.solved_files_question_text[ind_j].split()[:500])+"\n"+' '.join(bsl.solved_files_question_answer[ind_j].split()[:500])+"\n\n")
    different_pairs.close()    

if __name__ == '__main__':
    bsl = Baseline(1000)
    bsl.populate_data()
    similar_answers_different_questions(bsl)
    #answers_for_lucene(bsl)
    #questions_for_lucene(bsl)
    #auto_encoding_clustering(bsl)
    #sample_tf_idf_answer(bsl)
    #sample_tf_idf_question(bsl)
    #tf_idf_kmeans_answer(bsl)
    #tf_idf_kmeans_question(bsl)
    #tf_idf_kmeans_question(bsl)
    #tf_idf_question_question(bsl)
    #tf_idf_results(bsl)
    #tf_idf_question_answer(bsl)
    #random_50_pairs(bsl)
    #question_topic_counts(bsl)
    #run_answer_encoding = False
    #if len(sys.argv) > 1:
    #    run_answer_encoding = True
    #if run_answer_encoding:
    #    #answer_encoding_kmeans(bsl)
    #    #answer_encoding_label_kmeans(bsl)
    #    answer_encoding_magnet(bsl)
    #else:
    #    auto_encoding_clustering(bsl)
    #write_question_answer_text(bsl)
    #kmeans_clustering_encoding(bsl,num_clusters=400)
    #get_results(bsl)
    #tao_question_answer(bsl)
    #cluster_text(bsl,num_topics=50,use_answers=True,combine_question_answers=False,forTest=False)
    #cluster_text(bsl,num_topics=200,use_answers=False,combine_question_answers=False,forTest=False)
    #encoding_data_kmeans(bsl)
    #tao_question_answer(bsl)
    #cluster_text(bsl,num_topics=120,use_answers=False)
    #supportDocsUniqSubtitles(bsl)
    #getUniqSupportDocs(bsl)
    #getSupportDocsCosineSimilarities(bsl)
    #getSimilarPairs(bsl)
    #create_w2v_text(bsl)
    #create_basic_data(bsl)
    #create_data_model(bsl)
    #top_cosine(bsl)
    #create_encoding_data(bsl)
    #test_cosine(bsl)
    #tao_train_data_pairs(bsl,use_para=False,use_sentences=False,ignore_files=bsl.solved_files_with_support_docs)
    #tao_model_text(bsl)
    #train_encoding_data(bsl)
    #create_basic_data(bsl)
    #tao_model_toy_test_phrase_picker(bsl,toy_files=bsl.solved_files)
    #tao_model_text(bsl)
    #print get_best_sentences('../DATA/7622960','HT204323',[],bsl,use_sentences=False,use_tfidf=True)
    #print get_best_sentences('../DATA/7622960','HT204323',[],bsl,use_sentences=False,use_tfidf=False)
    #while True:
    #    ques_name = raw_input('Question name:')
    #    support_doc = raw_input('Support document id:')
    #    print get_best_sentences(ques_name,support_doc,[],bsl,use_sentences=False,use_tfidf=False)
    #tao_model_best_phrase(bsl,use_sentences=False,use_tfidf=False)
    #create_w2v(bsl)
    #lucene_text(bsl)
    #toy_files = ["../DATA/7622960","../DATA/7622175","../DATA/7622066","../DATA/7621761","../DATA/7621729"]
    #tao_model_text(bsl)
    #kfold(bsl,use_para=False,use_sentences=False,fold=4)
    #tao_train_data_pairs(bsl,use_para=True,use_sentences=False,ignore_files=bsl.solved_files_with_support_docs)
    #tao_test_data_pairs(bsl,use_para=True,use_sentences=False,ignore_files=list(set(bsl.solved_files)-set(bsl.solved_files_with_support_docs)))
    #print get_best_sentences(9552,'HT204264',[],bsl,use_sentences=True)
    #create_w2v(bsl)
    #lucene_text(bsl)
    #tao_model_text(bsl)
    #tao_train_data_pairs(bsl,use_para=True)
    #tao_test_data_pairs(bsl,use_para=True)
    #create_parallel_data(bsl)
    #copy_relevant_files(bsl)
    #get_dissimilar_words(bsl)
    #get_support_doc_cosine_sim(bsl)
    #get_lda_classes(bsl)
