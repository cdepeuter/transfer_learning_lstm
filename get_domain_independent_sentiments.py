
# coding: utf-8

# In[19]:


import pandas as pd
import os
import codecs
WORDS_TO_TAKE = 2000


# In[14]:


frames = []
for f in os.listdir("subreddits"):
    if f.endswith("tsv"):
        frame = pd.read_csv("subreddits/"+f, sep="\t", names=['word', 'sentiment', 'std_sentiment'])
        # only take strong polarity words
        frame = frame[(frame.sentiment>1) | (frame.sentiment < -1)]
        frame["reddit"] = f
        frames.append(frame)
        
data = pd.concat(frames)
data.head()


# In[15]:


print(data.shape)


# In[16]:


all_words = data.groupby('word').count()


# In[17]:


all_words.sort_values(by='sentiment', ascending=False)


# In[29]:


domain_independent_words = all_words.sort_values(by='sentiment', ascending=False).sentiment.index[0:WORDS_TO_TAKE]

with codecs.open("data/vocab/domain_independent_sentiment.txt", 'w', encoding='utf-8') as fp:
    for w in domain_independent_words:
        fp.write(w+"\n")

