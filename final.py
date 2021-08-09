import io
import os
import re
import csv
import nltk
import spacy
import pandas as pd
import docx2txt
from datetime import datetime
from dateutil import relativedelta
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFSyntaxError
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import json
import training
from spacy.matcher import Matcher
# import sys, fitz


def get_text_from_pdf(pdf_path):
    '''
    function to extract the plain text from .pdf files

    input param: (pdf_path) path to PDF file to be extracted
    return: iterator of string of extracted text
    ''' 

    if not isinstance(pdf_path, io.BytesIO):
   
        # extract text from pdf file
        with open(pdf_path, 'rb') as fh:
            try:
                for page in PDFPage.get_pages(
                        fh,
                        caching=True,
                        check_extractable=True
                ):
                    resource_manager = PDFResourceManager()
                    fake_file_handle = io.StringIO()
                    converter = TextConverter(
                        resource_manager,
                        fake_file_handle,
                        codec='utf-8',
                        laparams=LAParams()
                    )
                    page_interpreter = PDFPageInterpreter(
                        resource_manager,
                        converter
                    )
                    page_interpreter.process_page(page)

                    text = fake_file_handle.getvalue()
                    yield text

                    # close open handles
                    converter.close()
                    fake_file_handle.close()
            except PDFSyntaxError:
                return
    else:
        # extract text from pdf file
        try:
            for page in PDFPage.get_pages(
                    pdf_path,
                    caching=True,
                    check_extractable=True
            ):
                resource_manager = PDFResourceManager()
                fake_file_handle = io.StringIO()
                converter = TextConverter(
                    resource_manager,
                    fake_file_handle,
                    codec='utf-8',
                    laparams=LAParams()
                )
                page_interpreter = PDFPageInterpreter(
                    resource_manager,
                    converter
                )
                page_interpreter.process_page(page)

                text = fake_file_handle.getvalue()
                yield text

                # close open handles
                converter.close()
                fake_file_handle.close()
        except PDFSyntaxError:
            return

def get_text_from_docx(doc_path):
    '''
    function to extract plain text from .docx files

    input param : (doc_path) path to .docx file to be extracted
    :return: string of extracted text
    '''
    try:
        proccessed_text = docx2txt.process(doc_path)
        text = [line.replace('\t', ' ') for line in proccessed_text.split('\n') if line]
        return ' '.join(text)
    except KeyError:
        return ' '

def get_text_from_doc(doc_path):
    '''
    function to extract plain text from .do

    input param: (doc_path) path to .doc file to be extracted
    :return: string of extracted text
    '''

    try:
        try:
            import textract
        except ImportError:
            return ' '
        text = textract.process(doc_path).decode('utf-8')
        return text
    except KeyError:
        return ' '


def get_text(file_path, extension):
    '''
    function to detect the file extension and call text
    extraction function accordingly

    input param (file_path):  path of file of which text is to be extracted
    :param extension: extension of file `file_name`
    '''
    text = ''
    if extension == '.pdf':
        for page in get_text_from_pdf(file_path):
            text += ' ' + page
    elif extension == '.docx':
        text = get_text_from_docx(file_path)
    elif extension == '.doc':
        text = get_text_from_doc(file_path)
    return text

def get_total_experience(resume_text):
    '''
    function to extract total months of experience from a resume
    :param experience_list: list of experience text extracted
    :return: total months of experience
    '''
    WORDS=['fresher','1 year','2 years','3 years','4 years','5 years','6 years',
    '7 years','8 years','9 years','10 years','1year','2years','3years','4years','5years','6years',
    '7years','8years','9years','10years',
    '5+years','6+years',
    '7+years','8+years','9+years','10+years','5+ years','6+ years',
    '7+ years','8+ years','9+ years','10+ years']
    nlp = spacy.load('en_core_web_sm')

    # Grad all general stop words
    # STOPWORDS = set(stopwords.words('english'))
    nlp_text = nlp(resume_text)

    # Sentence Tokenizer
    nlp_text = [sent.text.strip() for sent in nlp_text.sents]
    l=[]
    for index, text in enumerate(nlp_text):
        for txt in text.split("\n"):
            for i in txt.split("\n"):
                for k in WORDS:
                    if k in i.lower():
                        if len(i.split(" "))>0:
                            l.append(i)
   
    if l:
        return l
    return 'NONE MENTIONED'
    

def get_experience(resume_text):
    '''
    function to extract experience from resume text

    input param(resume_text): Plain resume text
    return: list of experience
    '''
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # word tokenization
    word_tokens = nltk.word_tokenize(resume_text)

    # remove stop words and lemmatize
    filtered_sentence = [
            w for w in word_tokens if w not
            in stop_words and wordnet_lemmatizer.lemmatize(w)
            not in stop_words
        ]
    sent = nltk.pos_tag(filtered_sentence)

    # parse regex
    cp = nltk.RegexpParser('P: {<NNP>+}')
    cs = cp.parse(sent)

    # for i in cs.subtrees(filter=lambda x: x.label() == 'P'):
    #     print(i)

    test = []

    for vp in list(
        cs.subtrees(filter=lambda x: x.label() == 'P')
    ):
        test.append(" ".join([
            i[0] for i in vp.leaves()
            if len(vp.leaves()) >= 2])
        )

    # Search the word 'experience' in the chunk and
    # then print out the text after it
    
    x = [
        x[x.lower().index('experience') + 10:]
        for i, x in enumerate(test)
        if x and 'experience' in x.lower()
    ]
    return x

def get_skills(input_text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.tokenize.word_tokenize(input_text)

    # remove the stop words
    filtered_tokens = [w for w in word_tokens if w not in stop_words]

    # remove the punctuation
    filtered_tokens = [w for w in word_tokens if w.isalpha()]

    # generate bigrams and trigrams (such as artificial intelligence)
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))

    # we create a set to keep the results in.
    found_skills = set()
    with open('skills.txt') as fp:
        SKILLS_SET=fp.read().split(",")

    # we search for each token in our skills database
    for token in filtered_tokens:
        if token.lower() in SKILLS_SET:
            found_skills.add(token)

    # we search for each bigram and trigram in our skills database
    for ngram in bigrams_trigrams:
        if ngram.lower() in SKILLS_SET:
            found_skills.add(ngram)

    return list(found_skills)




def extract_education(resume_text):
    '''
    Helper function to extract education from spacy nlp text
    :param nlp_text: object of `spacy.tokens.doc.Doc`
    :return: tuple of education degree and year if year if found
             else only returns education degree
    '''
    nlp = spacy.load('en_core_web_sm')

    # Grad all general stop words
    STOPWORDS = set(stopwords.words('english'))

    EDUCATION = [
                'BE','B.E.', 'B.E', 'BS', 'B.S','C.A.','CA.','B.COM','BCOM','M.COM', 'MCOM',
             'M.E', 'MS', 'M.S', 'BTECH', 'B.TECH', 'M.TECH', 'MTECH',
            'PHD','PH.D.','MBA','GRADUATE', 'POST GRADUATE','5 YEAR INTEGRATED MASTERS',
            'SSC', 'HSC', 'CBSE', 'ICSE','BACHELOR',
            'INSTITUTE','UNIVERSITY','SCHOOL'
    
            ]
    nlp_text = nlp(resume_text)
    edu={}
    # Sentence Tokenizer
    nlp_text = [sent.text.strip() for sent in nlp_text.sents]
    # Extract education degree
    try:
        for index, text in enumerate(nlp_text):
            for tex in text.split():
                tex = re.sub(r'[?|$|.|!|,]', r'', tex)
                if tex.upper() in EDUCATION and tex not in STOPWORDS:
                    edu[tex] = text + nlp_text[index + 1]
    except IndexError:
        pass

    # Extract year
    education = []
    for key in edu.keys():
        year = re.search(re.compile(r'(((20|19)(\d{2})))'), edu[key])
        if year:
            education.append((key, ''.join(year.group(0))))
        else:
            education.append(key)
    return education




def extract_name(resume_text,matcher):
    NAME_PATTERN      = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    pattern = [NAME_PATTERN]
    matcher.add('NAME', None, *pattern)
    nlp_text = nlp(resume_text)

    # Sentence Tokenizer
    matches = matcher(nlp_text)
    
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text




def get_certifications(resume_text):
    WORDS=['CERTIFIED']
    nlp = spacy.load('en_core_web_sm')

    # Grad all general stop words
    # STOPWORDS = set(stopwords.words('english'))
    nlp_text = nlp(resume_text)

    # Sentence Tokenizer
    nlp_text = [sent.text.strip() for sent in nlp_text.sents]
    l=[]
    for index, text in enumerate(nlp_text):
        for txt in text.split("\n"):
            for i in txt.split("\n"):
                for k in WORDS:
                    if k in i.upper():
                        if len(i.split(" "))>0:
                            l.append(i)
   
    if l:
        return l
    return []

def extract_phone_number(resume_text):
    
    PHONE_REG = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
    phone = re.findall(PHONE_REG, resume_text)

    if phone:
        number = ''.join(phone[0])

        if resume_text.find(number) >= 0 and len(number) < 16:
            return number
    return None
  


def extract_emails(resume_text):
    EMAIL_REG = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')
    return re.findall(EMAIL_REG, resume_text)

import os
import spacy
nlp = spacy.load('en_core_web_sm')

def final_function(path):
    op=path
    if op.endswith(".pdf"):
        text =get_text(op,".pdf")
    if op.endswith(".docx"):
        text =get_text(op,".docx")
    
    d2=training.essentials(op)

    name2= extract_name(text,Matcher(nlp.vocab))

    name = d2["NAME"]
    location = d2["LOCATION"]
    skills = d2["SKILLS"]
    designation = d2["DESIGNATION"]


    d1={
    "YOE":get_total_experience(text),
    "PHONE": extract_phone_number(text),
     "EMAIL": extract_emails(text),
    "EDUCATION":extract_education(text),
    "SKILLS": list(set(get_skills(text)+skills)),
    "CERTIFICATIONS":get_certifications(text),
    "DESIGNATION":designation,
    "Location": list(set(location)),
    "NAME": name
    }

    print(d1)

    return json.dumps(d1)

# final_function('C:\\Users\\jahna\\Downloads\\IDEATHON\\FinalCode\\Aishwarya[5_0].pdf')
