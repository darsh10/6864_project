import operator
import py.test

from baseline import *

class CosineSimilarityBaseline(Baseline):
    def __init__(self, do_tf_idf, data_size, download_support_docs, scoring_n):
        super(CosineSimilarityBaseline, self).__init__(data_size=data_size, download_support_docs=download_support_docs,scoring_n=scoring_n)
        self.do_tf_idf = do_tf_idf
        self.scoring_n = scoring_n

    def do_cosine_similarity(self):
        map_support_docs_bow={}
        for doc, super_title in self.map_support_docs_super_titles.items():
            map_support_docs_bow[doc] = bow_vector(super_title.split(' '), self.all_words, binary=self.do_tf_idf)

        if self.do_tf_idf:
            bloblist=[]
            for i in range(len(self.solved_files_question_text[8000:])):
                text=self.solved_files_question_text[i+8000]
                bloblist.append(tb(text))
            print "Bloblist length: %d"%len(bloblist)

        num_correct=0
        for j, text in enumerate(self.solved_files_question_text[8000:]):
            doc_score_list = []

            i = j + 8000
            file_bow=bow_vector(text.split(' '), self.all_words, binary=self.do_tf_idf)
            if self.do_tf_idf:
                blob = bloblist[j]
                scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
                for word in scores.keys():
                    if word in self.all_words and self.all_words.index(word)>=0:
                        file_bow[self.all_words.index(word)]*=scores[word]

            cos_max=0.0
            cand_doc=""
            for doc in map_support_docs_bow.keys():
                doc_bow=map_support_docs_bow[doc]
                new_cos=compute_cos_sim(doc_bow,file_bow)
                if cos_max<new_cos:
                    cos_max=new_cos
                    cand_doc=doc
                doc_score_list.append((new_cos, cand_doc))

            actual_doc=self.solved_files_support_docs[self.solved_files[i]]
            if self.scoring_n == 1:
                if cand_doc==actual_doc:
                    num_correct+=1
            else:
                sorted_scores = sorted(doc_score_list, key=operator.itemgetter(0))[-self.scoring_n:]
                top_n_docs = set([i[1] for i in sorted_scores])
                if cand_doc in top_n_docs:
                    num_correct += 1
            if (j + 1) % 100 == 0:
                print
                print "Correctly identified documents so far: %d out of %d"%(num_correct, j+1)
        accuracy = num_correct * 1.0 / len(self.solved_files)
        print
        print "Final accuracy: %f"%accuracy
        return accuracy

def main():
    usage = "python cosine_similarity.py [-t to remove tf-idf] [-n <data size>] [-d to download missing support documents] [-k score top N results]"
    do_tf_idf = True
    data_size = sys.maxint
    download_support_docs = False
    scoring_n = 1
    try:
        opts, args = getopt.getopt(sys.argv[1:], "tn:dk:")
    except getopt.GetoptError:
        print usage
        sys.exit(1)
    for opt, arg in opts:
        if opt == "-t":
            do_tf_idf = False
        elif opt == "-n":
            data_size = int(arg)
        elif opt == "-d":
            download_support_docs=True
        elif opt == '-k':
            scoring_n = int(arg)
    if do_tf_idf:
        print "Using tf-idf."
    else:
        print "No tf-idf."
    if data_size < sys.maxint:
        print "Only using %d samples."%data_size
    else:
        print "Using full dataset."
    if download_support_docs:
        print "Will crawl missing support documents."
    else:
        print "Will not crawl missing support documents."
    b = CosineSimilarityBaseline(do_tf_idf, data_size, download_support_docs, scoring_n)
    b.populate_data()
    b.do_cosine_similarity()

if __name__ == "__main__":
    main()
