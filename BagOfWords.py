import nltk
import pandas as pd
import numpy as np
import re
import random
import string 

from bs4 import BeautifulSoup

from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import decomposition, pipeline, metrics, grid_search
from nltk.corpus import stopwords

from collections import Counter
####################################################################################


at_re = r'@'
hashtag_re = r'#'
bitly_re = r'bit\.ly.*\s?'
instagram_re = r'instagr\.am.*\s?'
url_re = r'https?:.*\s?'
tweeturl_re = r't\.co.*\s?'
pic_re = r'pic\.twitter\.com.+\s?'


def munger(text):
    """ Munger - given a dataframe, cleanup the text column
    
    Input: text (string) - Input data to be cleaned
    
    Return: new string that is cleaned up """
    #for (index, row) in data.iterrows():
    text = re.sub(at_re, "", text)
    text = re.sub(hashtag_re, "", text)
    text = re.sub(bitly_re, "", text)
    text = re.sub(instagram_re, "", text)
    text = re.sub(url_re, "", text)
    text = re.sub(tweeturl_re, "", text)
    text = re.sub(pic_re, "", text)
    #### set_value is considered the new preferred way of setting values
    #### It is also extremely fast when used with iterrows()
    #data.set_value(index,"text",text)
    #return data
    return text

def get_tokens(text):
    lowers = text.lower()
    no_punctuation = lowers.translate(None,string.punctuation)
    tokens=nltk.word_tokenize(no_punctuation)
    return tokens 

def stopword_filter(tokens):
    return [w for w in tokens if w not in stopwords.words('english')]

def get_sentiment_counts(tokens):
    num_pos = 0
    num_neg = 0
    num_neutral = 0

    for token in tokens:
        if token in positive_words:
            num_pos += 1
        elif token in negative_words:
            num_neg += 1
        else:
            num_neutral += 1
    
    return (num_pos, num_neg, num_neutral)

def get_sentiment_label(num_pos, num_neg, num_neutral):
    if num_pos > num_neg:
        return "pos"
    elif num_pos < num_neg:
        return "neg"
    else:
        return "neutral"

####################################################################################


### Reading data 
data = pd.read_csv ("Obama-2012-2011.csv",header=0, error_bad_lines =False)

### Read in negative words
negative_file =  open("negative-words.txt", "r")
negative_words = []
for line in negative_file.readlines():
    negative_words.append( line.strip() )
negative_file.close()
negative_words = set(negative_words)

### Read in positive words
positive_file =  open("positive-words.txt", "r")
positive_words = []
for line in positive_file.readlines():
    positive_words.append( line.strip() )
positive_file.close()
positive_words = set(positive_words)


### Parse Data
positive_counts = []
negative_counts = []
neutral_counts = []
sentiment_labels = []

threshold = 10
for (index, row) in data.iterrows():
    text = row['text']
    
    ### Cleanup text column
    text_munged = munger(text)
    
    ### Word tokenize the string
    text_tokens = get_tokens(text_munged)
    
    ### filter tokens by stopwords
    text_filtered_tokens = stopword_filter(text_tokens)
    
    ### Get sentiment counts
    (num_pos, num_neg, num_neutral) = get_sentiment_counts(text_filtered_tokens)
    
    ### Append counts to lists
    positive_counts.append(num_pos)
    negative_counts.append(num_neg)
    neutral_counts.append(num_neutral)
 
    #print("Tweet with text '{}' has pos {} and neg {}".format(text, num_pos, num_neg))
    
    ### Determine a sentiment label from the counts    
    sentiment_label = get_sentiment_label(num_pos, num_neg, num_neutral)
    sentiment_labels.append(sentiment_label)
    
    #if index > threshold: 
    #    break

### Add counts as new columns
data['positive'] = positive_counts
data['negative'] = negative_counts
data['neutral'] = neutral_counts
data['sentiment_label'] = sentiment_labels

#print(data[ ['text', 'positive', 'negative', 'sentiment_label'] ])
# print(data[ data['sentiment_label'] == 'pos' ]['text'])
print(data[ data['sentiment_label']])
