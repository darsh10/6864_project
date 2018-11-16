import sys
import os
from os import listdir
from os.path import isfile, join
from apple_support_docs import getSoup,getDocName
import pickle
import shutil

DOCS_PATH = '../docs_short/'
DATA_PATH = '../DATA_short/'
doc_files = list(sorted([join(DOCS_PATH,f) for f in listdir(DOCS_PATH) if isfile(join(DOCS_PATH,f))]))
data_files = set([join(DATA_PATH,f) for f in listdir(DATA_PATH) if isfile(join(DATA_PATH,f))])
map_file_doc = {}
uniq_docs = set()
for doc_file in doc_files:
    print doc_file
    user_file = doc_file[doc_file.rfind('/')+1:]
    user_file = user_file[:user_file.find('_')]
    if DATA_PATH+user_file not in data_files:
        continue
    soup = getSoup(doc_file)
    uniq_doc = getDocName(soup)
    if not uniq_doc:
        continue
    map_file_doc[user_file] = uniq_doc
    shutil.copy(DOCS_PATH+user_file+'_doc', DOCS_PATH+uniq_doc)
    uniq_docs.add(uniq_doc)

print "Total downloaded %d" %len(doc_files)
print "Number that are sensible %d" %len(map_file_doc)
print "%d map to %d uniq support docs" %(len(map_file_doc),len(uniq_docs))
pickle.dump( map_file_doc, open( "map_file_doc.p", "wb" ) )    
