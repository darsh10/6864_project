'''
Get information from apple HTML file
Can be made better using beautifulsoup
Hacky right now
'''

import re
import operator
import sys

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
