
import igraph
import pickle
import requests
import pickle
import requests
import opengraph
import pprint
import csv
import pylab
import requests
import opengraph
import twitter
import logging
import time
import jpype
import csv
import logging
import cairocffi as cairo 
from collections import defaultdict
import graphlab as gl
import pylab
import nltk
import re
import random
import string 
import prettytable
import prettytable
import textblob
import math
import re
from bs4 import BeautifulSoup
import csv 
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import decomposition, pipeline, metrics, grid_search
from nltk.corpus import stopwords
from collections import Counter




logger = logging.getLogger('crawler')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('crawler.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)


class TimeoutError(Exception):
    pass

def timeout(seconds=120, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator

@timeout()
def getFollowers(api, follower):

    return api.GetFriendIDs(follower)



### Twitter API

API_TOKENS = [
    {"consumer_key": '',
    "consumer_secret": '',
    "access_token_key": '-',
    "access_token_secret": '',
    "requests_timeout": 60},

    {"consumer_key": '',
    "consumer_secret": '',
    "access_token_key": '',
    "access_token_secret": '',
    "requests_timeout": 60}
]

apis = []
for token in API_TOKENS:
    apis.append( twitter.Api(consumer_key=token['consumer_key'],
                    consumer_secret=token['consumer_secret'],
                    access_token_key=token['access_token_key'],
                    access_token_secret=token['access_token_secret']))



account_screen_name = '@POTUS'
account_id = '1536791610'

nodes = set()
edges = defaultdict(set)


try:
    logger.info("Loading followers for %s" % account_screen_name)
    f = open("following1", "rb")
    following = pickle.load(f)

except Exception as e:
    logger.info("Failed. Generating followers for %s" % account_screen_name)
    following = api.GetFriendIDs(screen_name=account_screen_name)
    pickle.dump(following, open("following1", "wb"))

api_idx = 0
api = apis[api_idx]
api_updated = False
following_iter = range(len(following))


try:
    f = open("edges.follow2.dict", "rb")
    edges = pickle.load(f)
    logger.info("Loaded edges.follow2 into memory!")
except Exception as e:
    logger.info("Starting from SCRATCH: did not load edges.follow2 into memory!")
    pass

#main followers
for follower_index in following_iter:
    follower = following[follower_index]
    success = False

    # followers of the main followers 
    followers_depth2_list = []
    while not success:
        try:
            logger.info("Followers of Followers %s" % follower)
            followers_depth2_list = getFollowers(api, follower)
            success = True
        except TimeoutError as e:
            logger.info("Timeout after 60s for follower %d" % follower)
            success = True     
            continue
        except Exception as e:
            logger.info("API Exception %s; api-idx = %d" % (str(e), api_idx))
            
            if api_updated and api_idx % len(API_TOKENS) == 0 and api_idx >= len(API_TOKENS):
                logger.info("Save edges to pickle file for follower = %s" % follower)
                pickle.dump(edges, open("edges.follow2.dict", "wb"))
                logger.info("Sleeping ...")
                time.sleep(60)
                api_updated = False
            else:
                api_idx += 1
                api = apis[api_idx % len(API_TOKENS)]
                api_updated = True
            
    if followers_depth2_list:
        logger.info("Adding followers to the graph")
        edges[follower].update(nodes.intersection(followers_depth2_list))

display =  str(followers_depth2_list)
print display

logger.info("Save edges to pickle file for follower = %s" % follower)
pickle.dump(edges, open("edges.follow2.dict", "wb"))

import pickle
n = open("nodes.follow1.set", "rb")
nodes = pickle.load(n)

e = open("edges.follow2.dict", "rb")
edges = pickle.load(e)
22
f = open("following1", "rb")
following = pickle.load(f)


import logging
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


