#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 22:43:05 2021

@author: negarmaleki
"""

# Import Libraries
import pandas as pd

# Import Data
dataset = pd.read_csv('/home/n/negarmaleki/Downloads/Topic_modeling/reddit.csv')
final_data = pd.read_csv('/home/n/negarmaleki/Downloads/Topic_modeling/steemit.csv')
dataset.drop('Unnamed: 0', axis=1, inplace=True)
final_data.drop('Unnamed: 0', axis=1, inplace=True)
dataset = dataset.sample(n=1000, replace = False, random_state=42)
final_data = final_data.sample(n=1000, replace = False, random_state=42)
dataset.reset_index(inplace=True, drop=True)
final_data.reset_index(inplace=True, drop=True)

# Post Compile
doc_steemit = [final_data.iloc[doc,7] for doc in range(len(final_data))] 
steemit_label = ["steemit" for doc in range(len(final_data))]  
doc_reddit = [dataset.iloc[doc,6] for doc in range(len(dataset))]
reddit_label = ["reddit" for doc in range(len(dataset))]
doc_complete = doc_steemit + doc_reddit
label_complete = steemit_label + reddit_label
df = pd.DataFrame(doc_complete, columns=['document'])
df['platform'] = label_complete
df.reset_index(inplace=True, drop=True)

# Cleaning and Preprocessing
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('wordnet')

# For each row in doc_complete, we will read the text, tokenize it, remove stopwords, stem it
clean_doc = []
for text in doc_complete:
    text = re.sub(r'http://\S+|https://\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[!"#$%&()*+,-./:;<=>?[\]^_`{|}~]', ' ', text).lower()
    
    words= nltk.tokenize.word_tokenize(text)
    words = [w for w in words if w.isalpha()]
    words = [w for w in words if len(w)>2 and w not in stopwords.words('english')]
    
    ps = PorterStemmer()
    words = [ps.stem(w) for w in words]
    clean_doc.append(words)

vocab = []
for clean_string in clean_doc:
    for word in clean_string:
        if word not in vocab:
            vocab.append(word)

clean_doc_index = []
for clean_string in clean_doc:
    clean = []
    for word in clean_string:
        clean.append(vocab.index(word))
    clean_doc_index.append(clean)
        
for i in range(len(df)):
    if df.iloc[i,1] == "steemit":
        clean_doc_index[i] = [1,0,1] + clean_doc_index[i]
    else:
        clean_doc_index[i] = [0,1,1] + clean_doc_index[i]


sourceFile = open('our_data.txt', 'w')
for i in range(len(clean_doc_index)):
    for j in range(len(clean_doc_index[i])-1):
        sourceFile.write(str(clean_doc_index[i][j])+" ")
    sourceFile.write(str(clean_doc_index[i][j+1])+"\n")
sourceFile.close()

sourceFile_2 = open('our_vocabulary.txt', 'w')
for word in vocab:
    sourceFile_2.write(word+"\n")
sourceFile_2.close()



"""# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(clean_doc)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_doc]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=2, id2word = dictionary, passes=50)

print(ldamodel.print_topics(num_topics=2, num_words=3))"""
