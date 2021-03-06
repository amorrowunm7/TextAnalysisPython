import nltk
import pandas as pd
import numpy as np
import re
import random
import prettytable
import math
import re
from collections import defaultdict
import textblob
import pylab
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

filenameAFINN = 'AFINN-111.txt'
afinn = dict(map(lambda (w, s): (w, int(s)), [ 
            ws.strip().split('\t') for ws in open(filenameAFINN) ])) 

        
def sentimentAFINN(text):

    words = pattern_split.split(text.lower())
    sentiments = map(lambda word: afinn.get(word, 0), words)
    if  sentiments:
        sentiment = float(sum(sentiments))/math.sqrt(len(sentiments))
        
    else:
        sentiment = 0
    return sentiment

def sentimentDisplayValue(sentimentScore):
    if sentimentScore > 0.1:
        return "Positive" 
    elif sentimentScore < -0.1:
        return "Negative"
    else:
        return "Neutral"


####################################################################################


### Reading data 
allStatuses = pd.read_csv ("Obama.csv",header=0, error_bad_lines =False)

# AFINN-111 is as of June 2011 the most recent version of AFINN
            
# Word splitter pattern
pattern_split = re.compile(r"\W+")

# Create a pretty table of tweet contents and sentiment
pt = prettytable.PrettyTable(["Tweet Status", "Sentiment Score", "Sentiment"])
pt.align["Tweet Status"] = "l" 
pt.align["Sentiment Score"] = "l" 
pt.max_width = 60 
pt.padding_width = 1 # One space between column edges and contents (default)

totals = defaultdict(int)

for (index, row) in allStatuses.iterrows():
    text = row['text']
    text_munged = munger(text)
    sentimentScore = sentimentAFINN(text_munged)
    sentimentDisplay = sentimentDisplayValue(sentimentScore)
    totals[sentimentDisplay] = totals[sentimentDisplay] + 1
    pt.add_row([text_munged, sentimentScore,  sentimentDisplay])
    
print pt
print totals

# List of accounts to process, and our results dict
allStatuses = pd.read_csv ("Obama.csv",header=0, error_bad_lines =False)

from collections import Counter
from nltk.corpus import stopwords

words = { }
hashtags = { }
counters = { "words": { }, "hashtags": { } }


for (index, row) in allStatuses.iterrows():
    text = row['text']
    words[accounts] = [ w.lower() for t in allStatuses for w in t.text.split() if w.lower() not in stopwords.words('english') ]
    counters["words"][account] = Counter(words[account])
    hashtags[account] = [ hashtag.text.lower() for status in allStatuses[account] for hashtag in status.hashtags ]
    counters["hashtags"][account] = Counter(hashtags[account])

for account in counters["words"]:
    pt = prettytable.PrettyTable(field_names=['Word', 'Count'])
    [ pt.add_row(kv) for kv in counters["words"][account].most_common()[:20] ]
    pt.align['Word'], pt.align['Count'] = 'l', 'r' # Set column alignment
    print account
    print pt
    print
