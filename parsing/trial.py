import sys
import os
from os import listdir
from os.path import isfile, join
from apple_user_questions import getSoup,get_question_answer
DATA_PATH = "../DATA/"
start_files = 700000
end_files = 800000
discussion_files = list(sorted([join(DATA_PATH,f) for f in listdir(DATA_PATH) if isfile(join(DATA_PATH,f))]))[start_files:end_files]
solved = 0
solved_with_url = 0
list_urls = []
solved_files = []
for i,file in enumerate(discussion_files):
    if (i+1) % 20 == 0:
        print "For %d iterations solved %d, with url %d" %(i+1,solved,solved_with_url)
    soup  = getSoup(file)
    if not soup:
        continue
    t,b,a_c,a,l = get_question_answer(soup)
    if not a:
        continue
    solved += 1
    if l:
        if "support.apple.com" in l:
            solved_with_url += 1
            start_ind = l.find("http")
            end_ind = l[start_ind:].find("\"") + start_ind
            list_urls.append(l[start_ind:end_ind])
            solved_files.append(file)
    elif "http" in a and "support.apple.com" in a:
        solved_with_url += 1
        start_ind = a.find("http")
        list_urls.append(a[start_ind:start_ind+30])
        solved_files.append(file)

print "Total solved %d" %solved
print "Total solved with url  %d" %solved_with_url
file_urls = open("urls.txt_"+str(start_files)+"_"+str(end_files),"w")
for i,url in enumerate(list_urls):
    file_urls.write(solved_files[i]+"\n")
    file_urls.write(url+"\n")
file_urls.close()
