# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 12:18:55 2021

@author: Sundar
"""
import time
start_time = time.time()
import re
from bs4 import BeautifulSoup
import unicodedata
import pandas as pd
import tqdm
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import LancasterStemmer
ps = nltk.porter.PorterStemmer()
ls =nltk.stem.LancasterStemmer()
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')


shakes_file = open(r"t8.shakespeare.txt", "r+")
content = shakes_file.read() 
s1=pd.Series(content)
df1=pd.DataFrame(s1)

find_w_file = open(r"find_words.txt", "r+")
find = find_w_file.read() 


"""import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

TESTDATA1 = StringIO(content)

df1 = pd.read_csv(TESTDATA1, sep=";",header=None)"""

c=df1[0]

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_special_characters(text, remove_digits = False):
    patterns = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(patterns,"",text)
    return text

def simple_stemmers(text,stemmer = ps):
    text = " ".join([stemmer.stem(word)for word in text.split()])
    return text

#def expand_contraction(text):
#    return contraction.fix(text)

def spacy_lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ !='-PRON-' else word.text for word in text])


contractions_dict = {
    'didn\'t': 'did not',
    'don\'t': 'do not',
    "aren't": "are not",
    "can't": "cannot",
    "cant": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "didnt": "did not",
    "doesn't": "does not",
    "doesnt": "does not",
    "don't": "do not",
    "dont" : "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i had",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'm": "i am",
    "im": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they had",
    "they'd've": "they would have",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
    }

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))



def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

def remove_stopwords(text, is_lower_case=False, stopwords=None):
    if not stopwords:
        stopwords = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def text_pre_processor(text,html_strip=True, accented_char=True,contraction_expansion=True,text_lower_case=True,
                       text_stemming=False, text_lemmatization=True,special_char_removal=True, remove_digits=True,
                       stopword_removal=True, stopword_list=None): 
    #strip HTML
    if html_strip:
        text=strip_html_tags(text)
    
    #remove extra newlines(often might be present in really noisy text)
    text = text.translate(text.maketrans("\n\t\r"," "))

    #remove accented character
    if remove_accented_chars:
        text = remove_accented_chars(text)
    
   #expand contraction
    if contraction_expansion:
         text = expand_contractions(text)
    
   #Lemmatize text
    if text_lemmatization:
         text = spacy_lemmatize_text(text)
    
   #remove special characters and \or digits
    if remove_special_characters:
   #insert space between special characters to isolate them
        special_char_pattern = re.compile(r'([{.(-)!}])')
        text = special_char_pattern.sub("\\1 ", text)
        text = remove_special_characters(text, remove_digits = remove_digits)
    
   #stem text
    if text_stemming and not text_lemmatization:
         text = simple_stemmers(text)
    
   #lowercase the text
    if text_lower_case:
         text = text.lower()

   #remove stopwords
    if stopword_removal:
         text = remove_stopwords(text,is_lower_case = text_lower_case,stopwords=stopword_list)
    
   #remove extra whitespace
    text = re.sub(' +', ' ',text)
    text = text.strip()

    return text

def pre_process_corpus(docs):
    norm_docs = []
    for doc in tqdm.tqdm(docs):
        doc = strip_html_tags(doc)
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        doc = doc.lower()
        doc = remove_accented_chars(doc)
        doc = expand_contractions(doc)
        doc=remove_stopwords(doc)
        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()  
        norm_docs.append(doc)
  
    return norm_docs

x= pre_process_corpus(c)
text=pd.DataFrame(x)
text.columns=['Text']




import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

TESTDATA2 = StringIO(find)
df2 = pd.read_csv(TESTDATA2, sep=";",header=None)
df2.columns=['Words']


text_list = text.values.tolist()
find_words = df2.values.tolist()


text_list1=pd.DataFrame(text_list)
a=[]
for item in text_list1[0]:
       for i in item.split():
         #print(i)
         a.append(i)
    
dict=pd.read_csv("french_dictionary.csv",header=None)


dict1=pd.DataFrame(dict[1])
b=[]

for i in a:
#    print(i)
    for j in find_words:
        #print(j)
        if (i==j[0]):
         #print(i)
            b.append(i)
g=[]
e=[]          
for z in b:
   for k in dict[0]:
       if z in k:
                       
           e.append(z)
                            
                                          
#replacement
           for k,l in zip(dict[0],dict[1]):
               if (z==k):
                     # print(z)
                     d=z.replace(z,l)
                     g.append(d)
           
          
h=pd.DataFrame(e)            
f=pd.DataFrame(g)  

uniqueWords = [] 
for u in e:
      if not u in uniqueWords:
          uniqueWords.append(u);

print("\n\n\n (1) Unique list of words that was replaced with French words from the dictionary: \n")
print(uniqueWords)
  
print("\n\n\n (2) Number of times a word was replaced : \n")
print(h[0].value_counts())


print("\n\n\n (3) Frequency of each word replaced : \n")
print(f[0].value_counts())

print("\n\n\n (5) Time taken to process : \n")
print("--- %s seconds ---" % (time.time() - start_time))

import os, psutil
process = psutil.Process(os.getpid())
print("\n\n\n (6) Memory taken to process : \n")
print(process.memory_info().rss, "bytes")

