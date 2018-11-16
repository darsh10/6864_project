from baseline_new import *
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

class LuceneAnswerQuestions(Baseline):

    def set_lucene(self):
        lucene.initVM()
        analyzer = StandardAnalyzer(Version.LUCENE_4_9)
        reader = IndexReader.open(SimpleFSDirectory(File("lucene/answers_index/")))
        self.searcher = IndexSearcher(reader)
        self.qp = QueryParser(Version.LUCENE_4_9, "answer_text", analyzer)

    def get_top_for_query(self, query, top_n=10):
        query = self.qp.parse(QueryParser.escape(query))
        hits = self.searcher.search(query, top_n)
        if not hits:
            return []
        top_pred_list = []
        for j in range(min(top_n,len(hits.scoreDocs))):
            hit = hits.scoreDocs[j]
            pred = self.searcher.doc(hit.doc).get("answer_name")
            top_pred_list.append(pred)
        return top_pred_list

    def get_training_data(self):

        dev_file = open("dev.txt","w")
        test_file = open("test.txt","w")

        for ind,solved_file in enumerate(self.solved_files):
            if self.solved_files_question_label[ind].strip()=="":
                continue
            if ind%33 == 0 and ind%2!=0:
                solved_file = solved_file[solved_file.rfind('/')+1:]
                query = ' '.join((self.solved_files_question_answer[ind]+" "+self.map_support_docs_exact_title[self.solved_files_support_docs[self.solved_files[ind]]]).split()[:500])
                top_answers = self.get_top_for_query(query)
                indices = [_ for _ in range(len(self.solved_files))]
                random.shuffle(indices)
                for k in range(10):
                    answer_id = self.solved_files[indices[k]]
                    answer_id = answer_id[answer_id.rfind('/')+1:]+"_ans"
                    top_answers.append(answer_id)
                random.shuffle(top_answers)
                dev_file.write(solved_file+"_ans"+"\t"+' '.join(top_answers).strip()+"\n")
            if ind%66 == 0:
                solved_file = solved_file[solved_file.rfind('/')+1:]
                query = ' '.join((self.solved_files_question_answer[ind]+" "+self.map_support_docs_exact_title[self.solved_files_support_docs[self.solved_files[ind]]]).split()[:500])
                top_answers = self.get_top_for_query(query)
                indices = [_ for _ in range(len(self.solved_files))]
                random.shuffle(indices)
                for k in range(10):
                    answer_id = self.solved_files[indices[k]]
                    answer_id = answer_id[answer_id.rfind('/')+1:]+"_ans"
                    top_answers.append(answer_id)
                random.shuffle(top_answers)
                test_file.write(solved_file+"_ans"+"\t"+' '.join(top_answers).strip()+"\n")
        dev_file.close()
        test_file.close()



if __name__ == '__main__':
    data_size = sys.maxint
    download_support_docs=False
    b = LuceneAnswerQuestions(data_size=data_size, download_support_docs=download_support_docs)
    b.populate_data()
    b.set_lucene()
    b.get_training_data()
