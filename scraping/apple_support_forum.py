import re
import urllib
import os
from os import listdir
from os.path import isfile, join
from time import sleep
import random
import pickle

mypath = "../docs_short/"
if not os.path.exists(mypath):
    os.makedirs(mypath)

onlyfiles = set([f for f in listdir(mypath) if isfile(join(mypath,f))])
#Create instance of HTML parser
file_url = pickle.load( open( "file_url.p", "rb" ) )
ctr = 0
for file_name in sorted(file_url.keys())[7317:10318]:
    url_name = file_url[file_name]
    ctr = ctr+1
    if ctr%10 == 0:
        print "###"
        print
        print "Completed checking %d files" %ctr
        print
        print "###"
    sleep(5.0+2*random.random())
    file_name = file_name[file_name.rfind('/')+1:]
    url_particular = file_name+"_doc"
    if str(url_particular) in onlyfiles:
        print "Already gt %s" %(url_particular)
        continue
    print "Trying "+str(url_particular)
    thisurl = url_name
    handle = urllib.urlopen(thisurl)
    html_junk = handle.read().split('\n')
    file_data = ""
    for item in html_junk:
        item = item.rstrip('\n')
        item = item.lstrip()
        item = item.strip()
        item = re.sub('\n*','',item)
        file_data = file_data+"\n"+item
    print "File size received:",len(file_data.split('\n'))
    print url_name
    if len(file_data.split('\n')) < 400:
        print "Good chance of the file being an erroneous file"
        continue
    
    f = open(mypath+str(url_particular),'w')
    f.write(file_data)
    f.close()
    print "Completed ", (url_particular )

