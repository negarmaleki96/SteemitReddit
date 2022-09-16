#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 11:16:42 2021

@author: negarmaleki
"""

# Import Libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from langdetect import detect

nltk.download('stopwords')
nltk.download('wordnet')

# Import Steemit Dataset
data = pd.read_csv("/home/n/negarmaleki/Downloads/posts_dataset/posts_dataset_10percent.csv")
data.drop('Unnamed: 0', axis=1, inplace=True)

# Selecting category
data_category = data[(data.parent_permlink == 'cancer') | 
              (data.parent_permlink == 'diet' ) |
              (data.parent_permlink == 'healthy') |
              (data.parent_permlink == 'health') |
              (data.parent_permlink == 'yoga') |
              (data.parent_permlink == 'medical') |
              (data.parent_permlink == 'drugs') |
              (data.parent_permlink == 'medicine') |
              (data.parent_permlink == 'healthcare') |
              (data.parent_permlink == 'meditation')]
data_category.drop_duplicates(subset=['title'], inplace=True, ignore_index=True)

# Selecting english body
language = []
for i in range(len(data_category)):
    try:
        language.append(detect(data_category.iloc[i,7]))
    except:
        language.append("Not Detected")

dataset = pd.concat([data_category.reset_index(), pd.DataFrame(language, columns=['language'])], axis=1)
dataset.drop('index', axis=1, inplace=True)

list_en = []
for i in range(len(dataset)):
    if dataset.iloc[i,45] == 'en':
        list_en.append(dataset.iloc[i,:-1])

final_data = pd.DataFrame(list_en, columns=['id', 'author', 'permlink', 'category', 'parent_author',
       'parent_permlink', 'title', 'body', 'json_metadata', 'last_update',
       'created', 'active', 'last_payout', 'depth', 'children', 'net_rshares',
       'abs_rshares', 'vote_rshares', 'children_abs_rshares', 'cashout_time',
       'max_cashout_time', 'total_vote_weight', 'reward_weight',
       'total_payout_value', 'curator_payout_value', 'author_rewards',
       'net_votes', 'root_author', 'root_permlink', 'max_accepted_payout',
       'percent_steem_dollars', 'allow_replies', 'allow_votes',
       'allow_curation_rewards', 'beneficiaries', 'url', 'root_title',
       'pending_payout_value', 'total_pending_payout_value', 'active_votes',
       'replies', 'author_reputation', 'promoted', 'body_length',
       'reblogged_by'])

for i in range(len(final_data)):
    final_data.iloc[i,35] = str('https://steemit.com')+final_data.iloc[i,35]

payout = []
for i in range(len(final_data)):
    payout.append(float(final_data.iloc[i,23].split()[0])+float(final_data.iloc[i,24].split()[0]))

final_dataset = pd.concat([final_data.reset_index(), pd.DataFrame(payout, columns=['payout'])], axis=1)
final_dataset.drop('index', axis=1, inplace=True)

# Check for missing values
final_dataset['body'].isna().sum()
final_dataset['body'].fillna('missing', inplace=True)
input_data_steemit = final_dataset['body']

# For each row in input_data, we will read the text, tokenize it, remove stopwords, lemmatize it
new_body_steemit = []
for text in input_data_steemit:
    text = re.sub(r'http://\S+|https://\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[!"#$%&()*+,-./:;<=>?[\]^_`{|}~]', ' ', text).lower()
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    new_body_steemit.append(text)


# Import Reddit Dataset
dataset = pd.read_csv('/home/n/negarmaleki/Downloads/Reddit/df_reddit_1.csv')
dataset.drop('Unnamed: 0', axis=1, inplace=True)
dataset.drop_duplicates(subset=['title'], inplace=True, ignore_index=True)

# Drop is_video column
drop_ind = []
for i in range(len(dataset)):
    if int(dataset.iloc[i,2]) == 1:
        drop_ind.append(i)
dataset.drop(drop_ind, inplace=True)

# Selecting english body
language = []
for i in range(len(dataset)):
    try:
        language.append(detect(dataset.iloc[i,6]))
    except:
        language.append("Not Detected")

dataset = pd.concat([dataset.reset_index(), pd.DataFrame(language, columns=['language'])], axis=1)
dataset.drop('index', axis=1, inplace=True)

list_en = []
for i in range(len(dataset)):
    if dataset.iloc[i,11] != 'en':
        list_en.append(i)
dataset.drop(list_en, inplace=True)

# Check for missing values
dataset['selftext'].isna().sum()
dataset['selftext'].fillna('missing', inplace=True)
input_data_reddit = dataset['selftext']

# For each row in input_data, we will read the text, tokenize it, remove stopwords, lemmatize it
new_body_reddit = []
for text in input_data_reddit:
    text = re.sub(r'http://\S+|https://\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[!"#$%&()*+,-./:;<=>?[\]^_`{|}~]', ' ', text).lower()
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    new_body_reddit.append(text)


# Sample 1000 from Reddit and Steemit
steemit = pd.DataFrame(new_body_steemit, columns=['text'])
sample_steemit = steemit.sample(n=1000, replace=False)
sample_steemit['platform'] = 'Steemit'
sample_steemit.reset_index(inplace=True, drop=True)

reddit = pd.DataFrame(new_body_reddit, columns=['text'])
sample_reddit = reddit.sample(n=1000, replace=False)
sample_reddit['platform'] = 'Reddit'
sample_reddit.reset_index(inplace=True, drop=True)

# Concatenate Steemit and Reddit
df_st = pd.DataFrame(input_data_steemit)
df_st.columns = ['text']
df_re = pd.DataFrame(input_data_reddit)
df_re.columns = ['text']
df_1 = pd.concat([df_st, df_re], ignore_index=True)
df_1.to_csv('/home/n/negarmaleki/Downloads/Tone/df_originaltxt.csv')
df = pd.concat([sample_steemit, sample_reddit], ignore_index=True)
df.to_csv('/home/n/negarmaleki/Downloads/Tone/df1.csv')

# Watson Tone Analyzer
import json
from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
tone_analyzer = ToneAnalyzerV3(
    version='2017-09-21',
    authenticator=IAMAuthenticator('l3zdz3yvOhO_er8XDETfwaYS7Fv1aWjg_Pi_0N-nD0UU'))
tone_analyzer.set_service_url('https://api.us-south.tone-analyzer.watson.cloud.ibm.com/instances/87c84683-97c8-40b5-98a0-56121903f845')

tone = []
num_index = []
for i in range(len(df)):
    try:
        tone_analysis = tone_analyzer.tone(df.iloc[i,0]).get_result()
        #dict_tone = json.loads(json.dumps(tone_analysis, indent=2))
        #tone.append(dict_tone)
        tone.append(tone_analysis['document_tone'])
    except:
        num_index.append(i)
tone_1 = []
for i in range(len(tone)):
    tone_1.append(tone[i]['tones'])

tone_2 = []
tone_3 = []
for i in range(len(tone_1)):
    tone_2 = []
    if len(tone_1[i]) != 0:
        for j in range(len(tone_1[i])):
            tone_2.append(tone_1[i][j]['tone_name'])
    else:
        tone_2.append(None)
    tone_3.append(tone_2)
    
df_tone = pd.DataFrame(tone_3)
df_result = pd.concat([df, df_tone], axis=1, ignore_index=True)
df_result.columns = ['text', 'platform', 'tone_1', 'tone_2', 'tone_3', 'tone_4']
df_result.to_csv('/home/n/negarmaleki/Downloads/Tone/watson_tone1.csv')
#df_watson_tone.to_csv('/home/n/negarmaleki/Downloads/Steemit/df_steemit_watson_tone.csv')


# Import text2emotion Library
import text2emotion as te
texttoemo = []
for i in range(len(df)):
    text2emotion = te.get_emotion(df.iloc[i,0])
    texttoemo.append(text2emotion)

from ast import literal_eval
import numpy as np
df_texttoemo = pd.DataFrame(texttoemo)
df_result_2 = pd.concat([df, df_texttoemo], axis=1, ignore_index=True)
df_result_2.columns = ['text', 'platform', 'Happy', 'Angry', 'Surprise', 'Sad', 'Fear']
df_result_2.to_csv('/home/n/negarmaleki/Downloads/Tone/text2emotion_tone1.csv')

    
