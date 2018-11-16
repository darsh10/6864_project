import sys, pickle, collections, pdb
import lucene
 
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version as Version

f=open('lucene_corpus.txt','r')
text_map = {}
for line in f:
    name, text = line.split('\t')
    text_map[name] = text
f.close()

          
if __name__ == "__main__":
    lucene.initVM()
    indexDir = SimpleFSDirectory(File("index/"))
    writerConfig = IndexWriterConfig(Version.LUCENE_4_9, StandardAnalyzer(Version.LUCENE_4_9))
    writer = IndexWriter(indexDir, writerConfig)

    for name, text in text_map.items():
        doc = Document()
        doc.add(Field("doc_name", name, Field.Store.YES, Field.Index.ANALYZED))
        doc.add(Field("doc_text", text, Field.Store.YES, Field.Index.ANALYZED))
        writer.addDocument(doc)

    print "Closing index of %d docs..."%writer.numDocs()
    writer.close()
