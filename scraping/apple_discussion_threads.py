import re
import urllib
import os
from os import listdir
from os.path import isfile, join
from time import sleep
import random
urlText = []
    
mypath = "../DATA/"
if not os.path.exists(mypath):
    os.makedirs(mypath)
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
#Create instance of HTML parser
f_log = open("/tmp/apple_scrap_html.log","a")
for i in range(1000000):
    sleep(5.0+2*random.random())
    url_particular = 7100000 + i
    if str(url_particular) in onlyfiles:
        print "Already gt %d" %(url_particular)
    if i%1000 == 0:
        f_log = open("/tmp/apple_scrap_html_data.log","a")
        f_log.write(str(url_particular)+'\n')
        f_log.close()
    print "Trying "+str(url_particular)
    thisurl = "https://discussions.apple.com/thread/"+str(url_particular)
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
    count_try = 0
    if len(file_data.split('\n')) < 1080:
        for i in range(count_try):
            sleep(0.25)
            thisurl = "https://discussions.apple.com/thread/"+str(url_particular)
            #Feed HTML file into parser
            handle = urllib.urlopen(thisurl)
            html_junk = handle.read().split('\n')
            file_data = ""
            for item in html_junk:
                item = item.rstrip('\n')
                item = item.lstrip()
                item = item.strip()
                item = re.sub('\n*','',item)
                file_data = file_data+"\n"+item
            if len(file_data.split('\n')) >= 1080:
                break
    if len(file_data.split('\n')) < 1080:
        print "Good chance of the file being an erroneous file"
        continue
    print "Number of lines %d" %(len(file_data.split('\n')))
    f = open(mypath+str(url_particular),'w')
    f.write(file_data)
    f.close()
    print "Completed ", (url_particular )
    f_log = open("/tmp/apple_scrap_html_data.log","a")
    f_log.write(str(url_particular)+"\n")
    f_log.close()

f_log.close()
