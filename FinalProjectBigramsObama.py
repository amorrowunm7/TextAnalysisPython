


import logging
logging.getLogger("requests").setLevel(logging.WARNING)


# Lets load up the Twitter API
import twitter
import prettytable

api = twitter.Api(consumer_key='',
                consumer_secret='',
                access_token_key='-',
                access_token_secret='')

# Grab FAIR's tweet stream
#
# NOTE: do not include retweets, too many dupes (though for text analysis this might be a 
#      way to weigh more heavily text from tweets that are being retweeted by the account)
statuses = api.GetUserTimeline(screen_name='@realDonaldTrump', count=100, include_rts=False)

# Create a pretty table of tweet contents and any expanded urls
pt = prettytable.PrettyTable(["Tweet Status", "Expanded URLs"])
pt.align["Tweet Status"] = "l" # Left align city names
pt.align["Expanded URLs"] = "l" # Left align city names
pt.max_width = 60 
pt.padding_width = 1 # One space between column edges and contents (default)

# Add rows to the pretty table
for status in statuses:
    pt.add_row([status.text, "".join([url.expanded_url for url in status.urls]) ])

# Lets see the results!
print pt

allStatuses = pd.read_csv ("Donald.csv",header=0, error_bad_lines =False)

# List of accounts to process, and our results dict
accounts = ['@realDonaldTrump']
allStatuses = { }

# For each account, query tiwtter for top tweets
for account in accounts:
    allStatuses[account] = api.GetUserTimeline(screen_name=account, count=100, include_rts=False)

# Save results
import pickle
pickle.dump( allStatuses, open( "allStatuses", "wb" ) )

# Open content and accounts from previous week
allStatuses = pickle.load( open( "allStatuses", "rb" ) )
accounts = allStatuses.keys()

# Extract all the links from the crawled tweets
links = { }
for account in accounts:
    links[account] = [url.expanded_url  for status in allStatuses[account] for url in status.urls ]



content = {}
LIMIT = 50

# Try to open content from pickle file
try:
    content = pickle.load(open("allStatuses", "wb"))
except Exception as e:
    # Otherwise, loop thru all acounts
    for account in accounts:
        # Init account to have no content
        content[account] = {}
        print "Starting account %s" % account

        # Loop thru all links for the account
        for url in links[account][:LIMIT]:
            # Attempt to use boilerpipe's article extractor on url
            try:
                print " -- extracting %s with boilerpipe" % url
                extractor = Extractor(extractor='ArticleExtractor', url=url)
                content[account][url] = extractor.getText()
            except Exception as e:
                # If there is any issue, try using OpenGraph metadata
                try:
                    print " -- failed; extracting %s with og" % url
                    og =  opengraph.OpenGraph(url=url, scrape=True)
                    content[account][url] = "%s. %s" % (og.title, og.description)
                except Exception as e:
                    pass
    
    # Save results
    pickle.dump(content, open("allStatuses", "wb"))


import string

import nltk
from nltk.collocations import *
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

def preProcess(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    filtered_words = [w for w in tokens if not w in stopwords.words('english') and not w.isdigit()]
    print "preprocess:" + " ".join(filtered_words)
    return " ".join(filtered_words)

def displayBigrams(contentDict, threshold=1):
    temp = "".join([ contentDict[url] for url in contentDict])
    tokens = nltk.wordpunct_tokenize(preProcess(temp))
    print "tokens:" + str(tokens) 
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(threshold)
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    print "scored" + str(scored)
    return sorted([ (bigram, score) for (bigram, score) in scored ], key=lambda t: t[1], reverse=True)

display = displayBigrams(content['@realDonaldTrump'])
print "output:" + str(display)



