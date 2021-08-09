# from __future__ import unicode_literals, print_function
# import plac
import random
# from pathlib import Path
import spacy
# from tqdm import tqdm
# from spacy_langdetect import LanguageDetector
# from spacy.language import Language
# from spacy.training.example import Example
import pickle
import json
from spacy.gold import GoldParse
from spacy.scorer import Scorer


train_data = pickle.load(open('C:\\Users\\jahna\\Downloads\\IDEATHON\\FinalCode\\train_data.pkl', 'rb'))

# print(train_data[0])

nlp = spacy.blank('en')

examples=[
    ('Govardhana K Senior Software Engineer  Bengaluru, Karnataka, Karnataka - Email me on Indeed: indeed.com/r/Govardhana-K/ b2de315d95905b68  Total IT experience 5 Years 6 Months Cloud Lending Solutions INC 4 Month • Salesforce Developer Oracle 5 Years 2 Month • Core Java Developer Languages Core Java, Go Lang Oracle PL-SQL programming, Sales Force Developer with APEX.  Designations & Promotions  Willing to relocate: Anywhere  WORK EXPERIENCE  Senior Software Engineer  Cloud Lending Solutions -  Bangalore, Karnataka -  January 2018 to Present  Present  Senior Consultant  Oracle -  Bangalore, Karnataka -  November 2016 to December 2017  Staff Consultant  Oracle -  Bangalore, Karnataka -  January 2014 to October 2016  Associate Consultant  Oracle -  Bangalore, Karnataka -  November 2012 to December 2013  EDUCATION  B.E in Computer Science Engineering  Adithya Institute of Technology -  Tamil Nadu  September 2008 to June 2012  https://www.indeed.com/r/Govardhana-K/b2de315d95905b68?isid=rex-download&ikw=download-top&co=IN https://www.indeed.com/r/Govardhana-K/b2de315d95905b68?isid=rex-download&ikw=download-top&co=IN   SKILLS  APEX. (Less than 1 year), Data Structures (3 years), FLEXCUBE (5 years), Oracle (5 years), Algorithms (3 years)  LINKS  https://www.linkedin.com/in/govardhana-k-61024944/  ADDITIONAL INFORMATION  Technical Proficiency:  Languages: Core Java, Go Lang, Data Structures & Algorithms, Oracle PL-SQL programming, Sales Force with APEX. Tools: RADTool, Jdeveloper, NetBeans, Eclipse, SQL developer, PL/SQL Developer, WinSCP, Putty Web Technologies: JavaScript, XML, HTML, Webservice  Operating Systems: Linux, Windows Version control system SVN & Git-Hub Databases: Oracle Middleware: Web logic, OC4J Product FLEXCUBE: Oracle FLEXCUBE Versions 10.x, 11.x and 12.x  https://www.linkedin.com/in/govardhana-k-61024944/', [(1356, 1793, 'Skills')])

]

def train_model(train_data):
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last = True)
    
    for _, annotation in train_data:
        for ent in annotation['entities']:
            ner.add_label(ent[2])
            
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(100):
            print("Statring iteration " + str(itn))
            random.shuffle(train_data)
            losses = {}
            index = 0
            for text, annotations in train_data:
                try:
                    nlp.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        drop=0.2,  # drop
                        sgd=optimizer,  # callable to update weights
                        losses=losses)
                except Exception as e:
                    pass
                
#             print(losses)


# train_model(train_data)

# nlp.to_disk('nlp_model')
nlp_model = spacy.load('nlp_model')

# doc = nlp_model(train_data[0][0])



def accuracy_of_model(ner_model,examples):
    scorer = Scorer()
    example = []
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot)
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return scorer.scores


print(accuracy_of_model(nlp_model,examples))


# for ent in doc.ents:
#     print(f'{ent.label_.upper():{30}}- {ent.text}')

# fname = 'C:\\Users\\jahna\\Downloads\\IDEATHON\\FinalCode\\Aishwarya[5_0].pdf'

def get_text_from_docx(doc_path):
    import docx2txt

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

def essentials(fname):
    import sys, fitz
    if fname.endswith(".pdf"):
        doc = fitz.open(fname)
        text = ""
        for page in doc:
            text = text + str(page.getText())

        tx = " ".join(text.split('\n'))
    if fname.endswith(".docx"):
        tx = get_text_from_docx(fname)
        



    # # print(tx)

    doc = nlp_model(tx)
    
# print(tx)
    details={'NAME':[],'LOCATION':[],'DEGREE':[],'DESIGNATION':[],'SKILLS':[]}
    for ent in doc.ents:
        if ent.label_.upper()=='NAME': 
            details['NAME'].append(ent.text)

        if ent.label_.upper()=='LOCATION':
            details['LOCATION'].append(ent.text)

        if ent.label_.upper()=='DEGREE': 
    
            details['DEGREE'].append(ent.text)

        if ent.label_.upper()=='DESIGNATION':
            details['DESIGNATION'].append(ent.text)
        
        if ent.label_.upper()=='SKILLS':
            details['SKILLS'].append(ent.text)

    return details
        
# print(essentials())

# print(essentials('C:\\Users\\jahna\\Downloads\\IDEATHON\\FinalCode\\Aishwarya[5_0].pdf'))



