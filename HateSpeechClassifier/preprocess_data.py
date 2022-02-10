import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv # for excel too 
import json 
import random
import numpy as np
import itertools 
from sklearn.preprocessing import StandardScaler 

# opens json file and loads json as dictionary 
# on Windows encoding is utf8 so make sure to specify 
with open(conan_p, "r", encoding="utf8") as f:
    multi_target = json.load(f)

# print(davidson_p)
# print(multi_target)

davidson_full_data = pd.read_csv(davidson_p)
# print(davidson_full_data) 

# for ind in multi_target: 
# #     print(ind)
#     for head in multi_target[ind]:
# #         print(" " + head)
#         # print out 4 types (Hate speech, counter narrative, target person, version) and string value 
# #         print("  " + multi_target[ind][head])
#         multi_target[ind][head]

# Turns json file/python dict format into pandas dataframe 
mt_df = pd.DataFrame(multi_target) 
# display(mt_df) 

mt_df = mt_df.T # transpose to make tweets features 
display(mt_df) 

d_full_df = pd.DataFrame(davidson_full_data)
display(d_full_df)

# collects hate speech into dataframe and series 
hs_orig = mt_df.iloc[:, 0] # [0, :] # if not transposed
# print(hs_orig)

# collects counter narrative into dataframe and series 
# cn_orig = mt_df.iloc[:, 1] # [1, :] # if not transposed 
# print(cn_orig)

# collects target minority into dataframe and series 
min_tar_orig = mt_df.iloc[:, 2] # 2, :] # if not tranposed # pd.DataFrame(mt_df.iloc[2, :], columns = ["TARGET"])
# print(min_tar_orig)

orig_labels = np.unique(min_tar_orig)
print(orig_labels)

# add label for not hate speech 
labels = np.append(labels, 'none') 
print(labels)

# collects version of dataset into dataframe and series 
# vers_orig = mt_df.iloc[3, :] 
# print(vers_orig)

f.close()

stopwords = stopwords = nltk.corpus.stopwords.words("english")

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

stemmer = PorterStemmer()

#Get POS tags for tweets and save as a string
tweets = d_full_df.tweet
# tweet_tags = []
# for t in tweets:
#     tokens = basic_tokenize(preprocess(t))
#     tags = nltk.pos_tag(tokens)
#     tag_list = [x[1] for x in tags]
#     tag_str = " ".join(tag_list)
#     tweet_tags.append(tag_str)

davidson_class = d_full_df.iloc[:, -2] 
processed_class = davidson_class[davidson_class != 1] # excludes offensive speech 
processed_tweets = d_full_df[d_full_df['class'] != 1]
processed_tweets = processed_tweets[processed_tweets['class'] == 2]

processed_tweets = processed_tweets.iloc[:, -1]
print("Davidson processed tweets: ") 
print(processed_tweets)

combined_hate_list = hs_orig.append(processed_tweets) 
combined_hate_list = combined_hate_list.reset_index(drop = True)
# print(combined_hate_list) # numpy list for input  

print("Combined datasets: ") 
combined_hate_df = combined_hate_df.reset_index(drop = True) 
combined_hate_df = pd.DataFrame(combined_hate_df, columns = ["HATE_SPEECH"])

display(ind_combined_hate_df) 

# ind_combined_hate_df.to_excel('Combined_dataset.xlsx', index = False, encoding='utf8')
if not os.path.exists('combined_data.csv'):
    combined_hate_df.to_csv('combined_data.csv', sep = ",", index = False, encoding = 'utf8')

min_tar = min_tar_orig.replace(orig_labels, [i for i in range(len(orig_labels))])
print(min_tar)

processed_class = processed_class[processed_class == 2]
processed_class.replace({2 : 8}, inplace=True)
print(processed_class) # class 0 is hate speech and 2 is neither (2 becomes 8 or "none")

combined_tar_df = min_tar.append(processed_class).reset_index(drop = True) 
combined_tar_df = pd.DataFrame(combined_tar_df, columns = ["CLASS"])

display(combined_tar_df)

if not os.path.exists('combined_class.csv'):
    combined_tar_df.to_csv('combined_class.csv', sep = ",", index = False, encoding = 'utf8')

combined_dataset_df = pd.concat([ind_combined_hate_df, combined_tar_df], axis = 1)
combined_dataset_df = pd.DataFrame(combined_dataset_df, columns = ["HATE_SPEECH", "CLASS"])

if not os.path.exists('combined_dataset.csv'):
    combined_dataset_df = pd.concat([ind_combined_hate_df, combined_tar_df], axis = 1)
    combined_dataset_df.to_csv('combined_dataset.csv', sep = ",", index = False, encoding = 'utf8')

# print(tokens)
# print(tweet_tags)
# print(tag_list)
# print(tags)
    
def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    print(parsed_text)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    print(parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    print(parsed_text)
    return parsed_text

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]+", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]+", tweet.lower())).strip()
    return tweet.split()