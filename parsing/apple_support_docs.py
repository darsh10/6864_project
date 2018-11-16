import sys
reload(sys)
sys.setdefaultencoding('utf8')
import re
import os
from os import listdir
from os.path import isfile, join
import sys
from bs4 import BeautifulSoup
import string
from nltk.tokenize import sent_tokenize


def getDocName(soup):
    
    if not soup:
        return ""
    if not soup.find('link',{'rel':'canonical'}):
        return ""
    doc_name_text = str(soup.find('link',{'rel':'canonical'}))
    start_ind = doc_name_text.find('http')
    end_ind = doc_name_text[start_ind:].find("\"") + start_ind
    doc_url = doc_name_text[start_ind:end_ind]
    doc_name = doc_url[doc_url.rfind('/')+1:]
    return doc_name

def getSoup(fileName):
    if not os.path.isfile(fileName):
        return None
    f=open(fileName)
    #lines='\n'.join(f.readlines())
    lines = f.read()
    lines_new = []
    #for line in lines:
        #line = line.encode('ascii', 'ignore').decode('ascii')
        #lines_new.append(str(line.encode('utf-8')))
    f.close()
    soup = BeautifulSoup(lines, "html.parser", from_encoding="utf-8")
    return soup

def getTitle(soup):
    if soup.title:
        return clean_text(str(soup.title.string[:soup.title.string.rfind('-')]))
    return ""

def getSmallTitles(soup):
    return soup.findAll('h2')
    small_titles = soup.findAll('h2')
    small_titles_list = []
    for s in small_titles:
        small_titles_list.append(clean_text(str(s)))
    return small_titles_list

def clean_text(txt):
    punctuation_without_period = ''.join(set(string.punctuation) - set('.') - set('?'))
    replace_punctuation_without_period = string.maketrans(punctuation_without_period, ' '*len(punctuation_without_period))
    txt = ' '.join(txt.split())
    txt = re.sub('\n',' ',txt)
    txt = re.sub('\t',' ',txt)
    #return txt.encode('ascii', 'ignore').decode('ascii').rstrip().lstrip().lstrip('\t').rstrip('\t')
    txt = ' '.join(re.sub(r'[^\x00-\x7f]',r' ',txt).strip('\t').split())
    txt_sentences = sent_tokenize(txt)
    txt = ""
    for snt in txt_sentences:
        if snt and snt[-1] == '.':
            snt = snt[:-1].strip()
        txt = txt+" "+str(snt).translate(replace_punctuation_without_period)
    return ' '.join(txt.split()).strip()

def getText(soup):
    if not soup:
        return ""
    if not soup.find("h1", { "id" : "main-title" }):
        return ""
    title = str(soup.find("h1", { "id" : "main-title" }).text)
    subtitle = ""
    if soup.find("div", { "class" : "intro" }):
        subtitle = str(soup.find("div", { "class" : "intro" }).text)
    articleBody = ""
    if soup.find("div", { "id" : "sections" }):
        articleBody = str(soup.find("div", { "id" : "sections" }).text)
    date = ""
    if soup.find("time", { "itemprop" : "dateModified" }):
        date = str(soup.find("time", { "itemprop" : "dateModified" }).text)
    if title=="" and subtitle=="" and articleBody=="" and date=="":
        return ""
    text_title = clean_text(title) + ' ' + clean_text(subtitle)
    text_body = clean_text(articleBody) + ' ' + clean_text(date)

    return str(text_title + ' ' + text_body)

def getCompleteTextWithoutTitle(soup):
    if getText(soup):
        return getText(soup)[getText(soup).find(getTitleSubtitle(soup))+1:]
    return ""

def getTextWithoutTitle(soup):
    if not soup:
        return ""
    if not soup.find("h1", { "id" : "main-title" }):
        return ""
    title = str(soup.find("h1", { "id" : "main-title" }).text)
    subtitle = ""
    if soup.find("div", { "class" : "intro" }):
        subtitle = str(soup.find("div", { "class" : "intro" }).text)
    articleBodyCleanText = []
    articleBodyText = []
    articleBody = []
    if soup.find("div", { "id" : "sections" }):
        articleBodies = soup.find("div", { "id" : "sections" })
        for aB in articleBodies.find_all('section'):
            articleBody.append(aB.find_all('p'))
        for aB in articleBody:
            articleBodyText.append("")
            for k in aB:
                articleBodyText[-1] +=str(k.text)
                #articleBodyCleanText.append(clean_text(aB.text))
    for aB in articleBodyText:
        if clean_text(aB):
            articleBodyCleanText.append(clean_text(aB))
    return articleBodyCleanText

def getTextWithoutTitleSentences(soup):

    if not soup:
        return ""
    if not soup.find("h1", { "id" : "main-title" }):
        return ""
    title = str(soup.find("h1", { "id" : "main-title" }).text)
    subtitle = ""
    if soup.find("div", { "class" : "intro" }):
        subtitle = str(soup.find("div", { "class" : "intro" }).text)
    articleBodyCleanText = []
    articleBodyText = []
    articleBody = []
    if soup.find("div", { "id" : "sections" }):
        articleBodies = soup.find("div", { "id" : "sections" })
        for aB in articleBodies.find_all('p'):
            articleBodyText.append(aB.text)

    for aB in articleBodyText:
        if clean_text(aB):
            articleBodyCleanText.append(clean_text(aB))

    return articleBodyCleanText

def getParagraphTitlesBodies(soup):
    
    if not soup:
        return ""
    if not soup.find("h1", { "id" : "main-title" }):
        return ""
    title = str(soup.find("h1", { "id" : "main-title" }).text)
    subtitle = ""
    if soup.find("div", { "class" : "intro" }):
        subtitle = str(soup.find("div", { "class" : "intro" }).text)
    if not soup.find("div", { "id" : "sections" }):
        return [],[]
    sections = soup.find("div", { "id" : "sections" }).find_all('section')
    paragraph_titles = []
    paragraph_bodies = []
    for section in sections:
        if not section.text.strip():
            continue
        section_text = section.text
        section_title = ""
        if section.find('h2'):
            section_title = section.find('h2').text
        elif section.find('h3'):
            section_title = section.find('h3').text
        elif section.find('h4'):
            section_title = section.find('h4').text
        section_body = section_text[section_text.find(section_title)+len(section_title):]
        section_title = str(section_title)
        section_body = str(section_body)
        paragraph_titles.append(clean_text(section_title).lower())
        paragraph_bodies.append(clean_text(section_body).lower())
        #print paragraph_titles[-1]
        #print paragraph_bodies[-1]
    return paragraph_titles,paragraph_bodies

def getTitleSubtitle(soup):
    if not soup: 
        return ""
    if not soup.find("h1", { "id" : "main-title" }): 
        return ""
    title = str(soup.find("h1", { "id" : "main-title" }).text)
    subtitle = ""
    if soup.find("div", { "class" : "intro" }):
        subtitle = str(soup.find("div", { "class" : "intro" }).text)
    text_title = clean_text(title) + ' ' + clean_text(subtitle)
    return text_title

def getExactTitle(soup):
    if not soup:
        return ""
    if not soup.find("h1", { "id" : "main-title" }):
        return ""
    title = str(soup.find("h1", { "id" : "main-title" }).text)
    return clean_text(title).lower()

def getExactSubTitle(soup):
    if not soup:
        return ""
    if not soup.find("h1", { "id" : "main-title" }):
        return ""
    title = str(soup.find("h1", { "id" : "main-title" }).text)
    subtitle = ""
    if soup.find("div", { "class" : "intro" }):
        subtitle = str(soup.find("div", { "class" : "intro" }).text)
    return clean_text(subtitle).lower()

#import sys
#soup = getSoup(sys.argv[1])
#print getParagraphTitlesBodies(soup)
#print getDocName(soup)
#getParagraphTitlesBodies(soup)
#import IPython; IPython.embed()
#print getTextWithoutTitleSentences(soup)
#print getTextWithoutTitle(soup)
#print getText(soup)
#print getTitleSubtitle(soup)
