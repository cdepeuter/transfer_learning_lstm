import json
import re
import os
import sys
import codecs
import pandas as pd
from nltk.stem import WordNetLemmatizer
import collections

def review_sentiment(score):
    if score <= NEGATIVE_REVIEW_MAX:
        return 0
    elif score >= POSITIVE_REVIEW_MIN:
        return 1
    return -1

def clean_review_text(rev):
    """
        Lemmatize and lowercase everything, also remove punctuation
    """
    rev = rev.lower()
    rev = re.sub("(\.\.\.|\. |,!)" , " ", rev)
    rev = re.sub('[^-a-z0-9_ -]+', '', rev)
    tokens = rev.split()
    lemmatized_tokens = []
    for t in tokens:
        lemmatized_tokens.append(lemmatizer.lemmatize(t.strip(".,!'")))
        
    return ' '.join(lemmatized_tokens)



def clean_save_yelp(revs, biz):
    revs = revs[revs.stars == revs.stars.astype(int)]

    
    #business_map = {b:i for i, b in enumerate(rev 
    category_counts = collections.defaultdict(int)
    
    def add_all_cats(categories):
        if categories is not None and len(categories) > 0:
            for c in categories:
                category_counts[c] += 1
    
    biz.categories.map(add_all_cats)
    #category_counts
    top_cats = sorted(category_counts.items(), key=lambda x:x[1], reverse=True)
    NUM_CATEGORIES = 10
    cats = {c[0]:i for i, c in enumerate(top_cats[0:NUM_CATEGORIES])}
    # only keep a select subset of distinct categories
    def get_cat(biz_cat):
        for b in biz_cat:
            if b == 'Food' or b == 'Restaurants':
                return 0
            elif b == 'Shopping':
                return 1
            elif b == 'Beauty & Spas':
                return 2
            elif b == 'Nightlife':
                return 3
            elif b == 'Automotive':
                return 4
        
        return -1
    
        
    biz["cat"] = biz.categories.map(get_cat)

    revs = revs.merge(biz, left_on="business_id", right_on="business_id", )
    print(revs.columns)

    revs = revs[revs.cat > -1]

    # no half star reviews
    print("post star removal shape" ,revs.shape)
    good_reviews = revs[revs.cat > 0]
    good_reviews["stars"] = good_reviews["stars_x"]
    good_reviews["clean_text"] = good_reviews.text.map(clean_review_text)
    good_reviews["sentiment"] = good_reviews.stars.map(review_sentiment)
    good_reviews = good_reviews[["stars", "cat", "clean_text", "sentiment"]]
    print(good_reviews.shape)
    good_reviews.to_csv("./data/yelp/cleaned/yelp.csv", index=False)
    print("saved yelp reviews")

def clean_save_amazon(data_files):
    for f in data_files:
        datas = []
        if not os.path.isfile(f.replace(".json", ".csv")):
            print("processing %s" % f)
            with codecs.open(f) as fp:
                for l in fp:
                    rev = json.loads(l)
                    datas.append(rev)

            data = pd.DataFrame.from_records(datas)
            # only keep reviews with so many words
            data = data[data.reviewText.map(lambda x:len(x.split())) > 75]
            print("dataframe loaded, small reviews removed", data.shape)
            data["sentiment"] = data["overall"].map(review_sentiment)
            # remove middling reviews
            data = data[data.sentiment != -1]
            data["clean_text"] = data.reviewText.map(clean_review_text)

            # only keep columns we want
            data = data[["clean_text", "sentiment", "overall"]]
            file = f.replace("raw", "cleaned").replace(".json", ".csv")


            if data.shape[0] > 2*SPLIT_SIZE:
                print("splitting file", f)
                for d in range(int(data.shape[0]/SPLIT_SIZE)):
                    split_frame = data[d*SPLIT_SIZE:(d+1)*SPLIT_SIZE]
                    file = f.replace("raw", "cleaned").replace(".json", "_" + str(d) + ".csv")
                    print("saved", file)
                    split_frame.to_csv(file, index=False)
            else:
                data.to_csv(file, index=False)

            print("done with", f)
            print(data.shape)
        else:
            print("skipping", f)

if __name__ == '__main__':
    lemmatizer = WordNetLemmatizer()

    data_files = [f for f in os.listdir("data") if f.endswith(".json")]
    source = "amazon"
    SPLIT_SIZE = 65536
    NEGATIVE_REVIEW_MAX = 2
    POSITIVE_REVIEW_MIN = 4

    if len(sys.argv) > 1:
        source = sys.argv[1]
        SPLIT_SIZE = sys.argv[2]


    if source == "amazon":
        data_files = ["./data/amazon/raw/" + f for f in os.listdir("./data/amazon/raw/") if f.endswith("json")]
        clean_save_amazon(data_files)
    elif source == "yelp":
        with codecs.open("data/yelp/raw/yelp_training_set_review.json") as fp:
            datas = []
            for l in fp.readlines():
                datas.append(json.loads(l))
        reviews = pd.DataFrame.from_records(datas)
        
        with codecs.open("data/yelp/raw/yelp_training_set_business.json") as fp:
            datas = []
            for l in fp.readlines():
                datas.append(json.loads(l))
            biz_ = pd.DataFrame.from_records(datas)

        if reviews is not None and biz_ is not None:
            clean_save_yelp(reviews, biz_)
