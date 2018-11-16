'''
Get information from apple HTML file
Can be made better using beautifulsoup
Hacky right now
'''

import os
import re
import operator
import sys
from bs4 import BeautifulSoup
import string
from apple_support_docs import clean_text
from nltk.tokenize import sent_tokenize

def getSoup(fileName):
    if not os.path.isfile(fileName):
        return None
    f = open(fileName)
    lines = f.read()
    f.close()
    soup = BeautifulSoup(lines, "html.parser", from_encoding="utf-8")
    return soup

#convert an answer semi-html to the correct string
def get_answer_text(answer_data):
    data = ""
    start_index = answer_data.find('jive-rendered-content')+len('jive-rendered-content')+2
    return re.sub('<[^>]+>', '', answer_data[start_index:]).lstrip().rstrip()

#convert the answer's author from semi-html to the correct string
def get_answer_author(answer_data):
    author = ""
    start_index = answer_data.find('<span class=\"username\">"')+len('<span class=\"username\">"')
    return re.sub('<[^>]+>', '', answer_data[start_index:]).lstrip().rstrip()


#lot of important information from the chat html
def get_chat_data(fileName):
    f=open(fileName,"r")
    lines = f.readlines()
    f.close()
    topic = ""
    question_author = ""
    question_label = ""
    question_content = ""
    answer_authors = []
    answers = []
    answers_begin = 0
    solved_answer = ""
    is_solved = 0
    did_ask = 0
    question_solver = ""
    index_ctr = 0 
    asked_ques_string = ""
    asked_index = -1
    for line in lines:
        index_ctr+=1
        #line=line.rstrip().lstrip().rstrip('\n')
        if "j-inline-correct-answer" in line:
            is_solved=1
            question_solver=lines[index_ctr+8].rstrip()
        if "answer-marker" in line:
            answers_begin = 1
        if "span class=\"username\">" in line:
            answer_authors.append(get_answer_author(line))
        if "status-label" in line:
            topic=re.sub('<[^>]+>', '', line).lstrip().rstrip()
        if "q-marker" in line:
            line=re.sub('<[^>]+>', '', line).lstrip().rstrip()
            cur_index=line.find('Q:')+len('Q:')
            question_label=line[cur_index:].lstrip().rstrip()

        if "class=\"jiveTT-hover-user jive-username-link\"" in line:
            next_line = lines[lines.index(line)+1][1:]
            start_index = next_line.find("class=\"jiveTT-hover-user jive-username-link\">")+len("class=\"jiveTT-hover-user jive-username-link\">")
            question_author=next_line
            question_author = re.sub('<[^>]+>', '', question_author).lstrip().rstrip().lstrip('\n').rstrip('\n')
        if "jive-rendered-content" in line and len(answer_authors)==0 and is_solved==1:
            solved_answer=line
        if "jive-rendered-content" in line and len(answer_authors)==0 and question_label!="" and question_content=="":
            question_content = re.sub('<[^>]+>', '', line).lstrip().rstrip()
        if "jive-rendered-content" in line and answers_begin==1 and len(answer_authors)==len(answers)+1:
            answers.append(line)
            if did_ask==1:
                continue
            if solved_answer in answers:
                continue
            if answer_authors[-1]==question_solver and answer_authors[-1]!=question_author:
                #print answer_authors[-1],question_author
                if '?' in answers[-1]:
                    did_ask=1
                    asked_ques_string = answers[-1]
                    asked_index = len(answers)-1
                    #print answer_authors[-1],question_author
    if solved_answer=="":
        question_solver=""

    nonBreakSpace = u'\xa0'
    question_label = question_label.strip(nonBreakSpace)
    question_content = question_content.strip(nonBreakSpace)
    question_content = re.sub('&nbsp;','',question_content)
    return topic,question_author,question_label,question_content,answers,answer_authors,is_solved,get_answer_text(solved_answer),did_ask,question_solver,get_answer_text(asked_ques_string),asked_index

def get_question_answer(soup):
    
    question_text = ""
    question_answer = ""
    question_answer_link_referred = ""
    question_title = ""
    if soup.find("section", { "class" : "j-original-message" }):
        question_text = soup.find("section", { "class" : "j-original-message" }).text
        question_text = question_text[:question_text.find("var textMore =")].strip()
        question_title = question_text[2:question_text.find("\n\n\n")].strip()
        question_text = question_text[question_text.find(question_title)+len(question_title)+3:].strip()
    solved_answer_date = ""
    if soup.find("div", { "class" : "j-inline-correct-answer" }):
        question_answer = soup.find("div", { "class" : "j-inline-correct-answer" }).text
        if soup.find("div", { "class" : "j-inline-correct-answer" }).find('section'):
            question_answer_link_referred = str(soup.find("div", { "class" : "j-inline-correct-answer" }).find('section').find('a'))
        question_answer = question_answer[question_answer.find("A:")+2:question_answer.find("Posted on ")].strip()
    
    question_text_list = sent_tokenize(question_text)
    question_text_sent_list = []
    solved_answer_date = "bhah blah"
    if soup.find("div", { "class" : "j-inline-correct-answer" }).find("p", {"class": "meta-posted"}):
        solved_answer_date = soup.find("div", { "class" : "j-inline-correct-answer" }).find("p", {"class": "meta-posted"}).text
        solved_answer_date = solved_answer_date[solved_answer_date.find("Posted On ")+len("Posted On "):]
    question_text_list = sent_tokenize(question_text)
    question_text_sent_list = []
    question_author = "whatdafuq"
    if soup.find("div",{"class":"j-post-author"}):
        if soup.find("div",{"class":"j-post-author"}).find("img"):
            question_author = soup.find("div",{"class":"j-post-author"}).find("img").text.strip()
    replier_authors = []
    for author in soup.find_all("span",{"class":"username"}):
        if author:
            replier_authors.append(author.text.strip())
    author_reply_index = len(replier_authors)+1000
    for k,author in enumerate(replier_authors):
        if author == question_author:
            author_reply_index = k+1
            break
    for s in question_text_list:
        question_text_sent_list.append(str(clean_text(s)))
    question_answer_clean = str(clean_text(question_answer))
    question_title = str(clean_text(question_title))
    question_text = str(clean_text(question_text))
    #return question_title,question_text,question_answer_clean,question_answer,question_answer_link_referred,question_text_sent_list #do something about this shit later
    question_topic = ""
    if soup.body:
        if soup.body.find("nav",{"id":"jive-breadcrumb"}):
            question_topic = soup.body.find("nav",{"id":"jive-breadcrumb"}).find_all('a')[-2].text
    return question_title,question_text,question_answer_clean,question_answer,question_answer_link_referred,question_text_sent_list,question_topic
    num_responses = 0
    corr_resp = -1
    if soup.find("span", { "class" : "j-inresponse-to" }):
        num_responses = len(soup.find_all("span", { "class" : "j-inresponse-to" }))
        all_answers = soup.find_all("span", { "class" : "j-inresponse-to" })
        for k,part_ans in enumerate(all_answers):
            ans_date = ""
            if part_ans.find("span", {"class":"u-hide--small"}):
                 ans_date = part_ans.find("span", {"class":"u-hide--small"}).text
            if str(ans_date).strip() == str(solved_answer_date).strip():
                 corr_resp = k+1
                 break
    if corr_resp == -1:
        corr_resp = num_responses+1
    if corr_resp != num_responses+1:
        corr_resp -=1
    if corr_resp == num_responses+1:
        corr_resp = -1
    possible_clarification = 0
    if author_reply_index < (corr_resp+1):
        possible_clarification = 1
    all_replies = []
    all_contents = soup.find_all("div",{"class":"jive-rendered-content"})[1:]
    all_contents_uniq = []
    if possible_clarification:
        author_reply_index -=1
    else:
        author_reply_index = -1
    for content in all_contents:
        if clean_text(content.text) not in all_contents_uniq:
            all_contents_uniq.append(clean_text(content.text))
        else:
            all_contents_uniq.remove(clean_text(content.text))
            all_contents_uniq.append(clean_text(content.text))
    intg_reply = -1
    for i,author in enumerate(replier_authors):
        if i==corr_resp:
            break
        if corr_resp>0 and corr_resp < len(replier_authors) and author == replier_authors[corr_resp]:
            intg_reply = i
    for i,author in enumerate(replier_authors):
        if intg_reply!=-1 or i==corr_resp:
            break
        if author !=question_author:
            intg_reply = i
    num_replies_till_correct = 0
    num_replies_till_correct = corr_resp+1
    print "How many am I returning "
    return question_title,question_text,question_answer_clean,question_answer,question_answer_link_referred,question_text_sent_list,question_topic,num_responses,num_replies_till_correct,possible_clarification,all_contents_uniq,author_reply_index,question_author,intg_reply,replier_authors

#len(soup.find_all("span", { "class" : "j-inresponse-to" }))
#soup = getSoup(sys.argv[1])
#print len(soup.find_all("span", { "class" : "j-inresponse-to" }))
#print get_question_answer(soup)
#import IPython; IPython.embed()
#soup.find("span", { "class" : "j-inresponse-to" }).find("span", {"class":"u-hide--small"}).text
#soup.find("div", { "class" : "j-inline-correct-answer" }).find("p", {"class": "meta-posted"}).text	
#soup.find_all("span",{"class":"username"})
#soup.find("div",{"class":"j-post-author"}).find("img").text
