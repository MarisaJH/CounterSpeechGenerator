import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv # for excel too 
import json 
import random
import numpy as np
import itertools 
from sklearn.preprocessing import StandardScaler 

# visualizing which of the word is most commonly used in the twitter dataset
from wordcloud import WordCloud

# interpolation - https://matplotlib.org/3.2.1/gallery/images_contours_and_fields/interpolation_methods.html
all_words = ' '.join([text for text in mt_df_T['HATE_SPEECH'] ])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

all_words2 = ' '.join([preprocess(text) for text in processed_tweets])
wordcloud2 = WordCloud(width = 800, height = 500, random_state = 0, max_font_size = 110).generate(all_words2)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud2, interpolation="bilinear")
plt.axis('off')
plt.show()