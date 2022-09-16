#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:48:22 2021

@author: negarmaleki
"""
# import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
#import h5py
#from gensim.models import Word2Vec
import gensim.models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
#from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
#import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense, Dropout, Activation, Bidirectional
from keras.layers import LSTM, SimpleRNN, GRU
import re
from nltk.tokenize import word_tokenize
#import preprocess
import nltk
from nltk.corpus import stopwords 
#from nltk.tokenize import word_tokenize 
nltk.download('stopwords')
nltk.download('punkt')

df_1 = pd.read_csv('/home/n/negarmaleki/Downloads/Clickbait/trainset_label.csv')
df_1.drop_duplicates(subset=['headline'], inplace=True)
df_1.reset_index(drop=True, inplace=True)

# clickbait headlines
df_2 = []
for i in range(len(df_1)):
    if df_1.iloc[i,1] == 1:
        df_2.append(df_1.iloc[i,:])
df_2 = pd.DataFrame(df_2, columns=['headline', 'label'])

# non-clickbait headlines
df_3 = []
for i in range(len(df_1)):
    if df_1.iloc[i,1] == 0:
        df_3.append(df_1.iloc[i,:])
df_3 = pd.DataFrame(df_3, columns=['headline', 'label'])

# balancing datasets (equal number of clickbaits and non-clickbaits in training set)
newdf_0 = pd.DataFrame(np.repeat(df_3.values, 2, axis=0), columns=df_3.columns)
newdf_0=newdf_0.sample(frac=1)
newdf_0.reset_index(drop=True, inplace=True)
newdf_1 = pd.DataFrame(np.repeat(df_2.values, 11, axis=0), columns=df_2.columns)
newdf_1=newdf_1.sample(frac=1)
newdf_1.reset_index(drop=True, inplace=True)

# final training dataset
df = pd.concat([newdf_0,newdf_1], ignore_index=True)
df=df.sample(frac=1)
df.reset_index(drop=True, inplace=True)

final_data = pd.read_csv('steemit_test.csv')
dataset = pd.read_csv('reddit_test.csv')
final_data.drop('Unnamed: 0', axis=1, inplace=True)
dataset.drop('Unnamed: 0', axis=1, inplace=True)

steemit = final_data[['title']]
steemit.columns = ['headline']
steemit['label'] = 2
reddit = dataset[['title']]
reddit.columns = ['headline']
reddit['label'] = 2
df_4 = pd.concat([steemit,reddit])
df_4.reset_index(drop=True, inplace=True)

# Import word2vec google
WORD2VEC_VECTORS_BIN = '/home/n/negarmaleki/GoogleNews-vectors-negative300.bin'

w2v = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_VECTORS_BIN, binary=True)
stop_words = set(stopwords.words('english'))

##Data cleaning
# Utils
from bs4 import BeautifulSoup

def tokenize_tweet(tweet):
    """
    Return a list of cleaned word tokens from the raw review
    """
    # Remove any HTML tags and convert to lower case
    text = BeautifulSoup(tweet).get_text()

    # Remove links
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    # Remove RTs

    if text.startswith('RT'):
        text = text[2:]

    text = re.sub('[\W_]+', ' ', text)
    text = text.strip()
    words = [w for w in word_tokenize(text.lower()) if w != 's']
    return words

def clean_tweet(tweet):
    """
    Return a list of cleaned word tokens from the raw review
    """
    # Remove any HTML tags and convert to lower case
    text = BeautifulSoup(tweet).get_text()

    # Remove links
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    # Remove RTs

    if text.startswith('RT'):
        text = text[2:]

    text = re.sub('[\W_]+', ' ', text)
    text = text.strip().lower()
    return text

# train and test concatenate
df_total = pd.concat([df, df_4], ignore_index=True)
df = df_total
df['cleaned_tweet'] = df['headline'].apply(clean_tweet)
df_4['cleaned_tweet'] = df_4['headline'].apply(clean_tweet)
#df = df.sample(frac=1)
#df.reset_index(drop=True, inplace=True)

# to find the maximum number of words in a sentence in order to determine the sequence size
L=[]
for i,token in enumerate(df['cleaned_tweet']):
    word=[w for w in token.split() if not w in stop_words]
    L.append(len(word))
    
sequence_size=max(L)

# separate the data in a non-random way to avoid having a problem during the cross validation due to the indexes
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df.iloc[:2028,1])
text_train,text_test,y_train,y_test=train_test_split(df.iloc[:2028,2], y,
                                                    stratify=y, 
                                                    test_size=0.15)
#df.iloc[:split,2],df.iloc[split:,2],df.iloc[:split,1],df.iloc[split:,1]

# Apply the word2vec from google but MOT by MOT !! the problem is that not all the words will appear in the word2vec
dimsize=300
def compute_matrix(text):
    X=np.zeros((len(text),sequence_size,dimsize))
    for i,token in enumerate(text):
        word=token.split()
        try:
            j=0
            for w in word:
                if w not in stop_words:
                    X[i,j]=w2v[w]
                    j+=1
        except: 
            pass
    return X 

a=pd.DataFrame(text_test.tolist(),columns=['headline'])
a['label']= pd.Series(y_test).tolist()
b = pd.DataFrame(df_4['cleaned_tweet'].tolist(), columns=['headline'])
b['label'] = df_4[['label']]
df_test = pd.concat([a,b],ignore_index=True)
text_test = df_test.iloc[:,0]
y_test = df_test.iloc[:,1]
X_train=compute_matrix(text_train)
X_test=compute_matrix(text_test)

# Create the model
def creat_model():
    
    model = Sequential()
    model.add(Bidirectional(SimpleRNN(128, dropout=0.2, recurrent_dropout=0.2), input_shape=(sequence_size, dimsize)))  # try using a GRU and a SimpleRNN
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    
    return model

# Define a file which saves the weights of the best model 
cp = ModelCheckpoint("best_model.h5",verbose=1,save_best_only=True)
earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto')
 
callback_list=[cp,earlystop_cb]  
batch_size=8
# Try using different optimizers and different optimizer configs
def fit_model(model,X_train,y_train,x_valid,y_valid,batch_size=batch_size):
    
    #earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
    
    model.fit(X_train, y_train, batch_size=batch_size, epochs=100, validation_data=(x_valid,y_valid), callbacks=callback_list)
    #score, acc = model.evaluate(X_test, y_test,batch_size=batch_size)
    return model

# Cross_validation 10folds

n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

for i, (train, valid) in enumerate(skf.split(X_train,y_train)):
    print(X_train[train].shape,y_train[train].shape,X_train[valid].shape,y_train[valid].shape)
    print ("Running Fold", i+1, "/", n_folds)
    model = None # Clearing the NN.
    model = creat_model()
    hist=fit_model(model, X_train[train], y_train[train], X_train[valid], y_train[valid])

# Save the weights of the best model to use them in the test
hist.load_weights("best_model.h5")
#model = tf.keras.models.load_model("best_model.h5")
# EVALUATION
f = open('model85-15.txt', 'w')
"""score, acc = hist.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score, file = f)
print('Test accuracy:', acc, file = f)"""

# Prediction
predict_x = model.predict(X_test)
y_pred = np.where(predict_x >= 0.5, 1, 0)
y_scores = model.predict(X_test)
y_pred_df = pd.DataFrame(y_pred,columns=['y_pred'])
#y_pred_df['y_score'] = y_scores
y_pred_df['headline'] = text_test.tolist()
#y_pred_df['platform'] = plat
y_pred_df['our_label'] = y_test.tolist()
#y_pred_df['original_headline'] = df_2[['headline']]
y_pred_df.to_csv('model85-15.csv')

print('Total Clickbaits = ', sum(y_pred), file = f)
print('\n', file = f)
print('Clickbait Percentage = ', sum(y_pred)/len(y_pred), file = f)

"""##afficher les résultats de la prediction
roc = roc_auc_score(y_test, y_scores)
print('ROC score:', roc, file = f)
# 
metrics = classification_report(y_test, y_pred, digits=4)
print('Classification Report \n', file = f)
print(metrics, file = f)

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix \n', file = f)
print(cm, file = f)"""
f.close()