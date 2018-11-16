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

class LuceneSimilarQuestions(Baseline):

    def set_lucene(self):
        lucene.initVM()
        analyzer = StandardAnalyzer(Version.LUCENE_4_9)
        reader = IndexReader.open(SimpleFSDirectory(File("lucene/questions_index/")))
        self.searcher = IndexSearcher(reader)
        self.qp = QueryParser(Version.LUCENE_4_9, "question_text", analyzer)

    def get_top_for_query(self, query, top_n=100):
        query = self.qp.parse(QueryParser.escape(query))
        hits = self.searcher.search(query, top_n)
        if not hits:
            return []
        top_pred_list = []
        for j in range(min(top_n,len(hits.scoreDocs))):
            hit = hits.scoreDocs[j]
            pred = self.searcher.doc(hit.doc).get("question_name")
            top_pred_list.append(pred)
        return top_pred_list

    def get_training_data(self):
        
        clusters = pickle.load(open("clusters.p","rb"))
        index_clusters = {}
        name_clusters = {}
        for label in clusters:
            for ind in clusters[label]:
                index_clusters[ind] = label
                name = self.solved_files[ind]
                name = name[name.rfind('/')+1:]
                name_clusters[name] = label

        train_file = open("train.txt","w")
        test_file = open("test.txt","w")

        for ind,solved_file in enumerate(self.solved_files):
            
            solved_file = solved_file[solved_file.rfind('/')+1:]
            query = ' '.join((self.solved_files_question_label[ind]+" "+self.solved_files_question_content[ind]).split()[:500])
            if (ind+1)%100==0:
                print "Got to index %d" %(ind+1)
            if query.strip() == "":
                continue
            near_neigbours = self.get_top_for_query(query)
            near_set = set(near_neigbours)
            pos = []
            negatives = []
            if ind%250 != 0:
                for j,solved_file_j in enumerate(self.solved_files):
                    if ind==j:
                        continue
                    solved_file_j = solved_file_j[solved_file_j.rfind('/')+1:]
                    if solved_file_j not in near_set:
                        negatives.append(solved_file_j)
                        continue
                for neighbour in near_neigbours:
                    if neighbour == solved_file:
                        continue
                    if name_clusters[solved_file] != name_clusters[neighbour]:
                        continue
                    pos.append(neighbour)
                    break
                train_file.write(solved_file+"\t"+' '.join(pos).strip()+"\t"+' '.join(negatives).strip()+"\n")
            else:
                near_neigbours = self.get_top_for_query(query,500)
                for neighbour in near_neigbours:
                    if neighbour == solved_file:
                        continue
                    if name_clusters[neighbour] == name_clusters[solved_file]:
                        pos.append(neighbour)
                    else:
                        negatives.append(neighbour)
                test_file.write(solved_file+"\t"+' '.join(pos).strip()+"\t"+' '.join(negatives).strip()+"\n") 
        train_file.close()
        test_file.close()

if __name__ == '__main__':
    data_size = sys.maxint
    download_support_docs=False
    b = LuceneSimilarQuestions(data_size=data_size, download_support_docs=download_support_docs)
    b.populate_data()
    b.set_lucene()
    b.get_training_data()
