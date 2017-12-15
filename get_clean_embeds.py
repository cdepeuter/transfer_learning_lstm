import gensim
import json
import codecs
import pandas as pd
import numpy as np
import re
import os
import sys
import collections
from imblearn.over_sampling import SMOTE
import sklearn

def get_vocab(name="regular_vocab"):
    with codecs.open("data/vocab/"+name+".json", encoding='utf-8') as f:
        vocab = json.loads(f.read())

    with codecs.open("data/vocab/"+name+"_lookup.json", encoding='utf-8') as f:
        lookup= json.loads(f.read())

    return vocab, lookup


def one_hot_rating(label):
    if label == 1:
        return np.array([1,0,0,0,0])
    elif label == 2:
        return np.array([0,1,0,0,0])
    elif label == 3:
        return np.array([0,0,1,0,0])
    elif label == 4:
        return np.array([0,0,0,1,0])
    elif label == 5:
        return np.array([0,0,0,0,1])
# yeah this is a bad way to do it oh well
def one_hot_category(label):
    if label == 0:
        return np.array([1,0,0,0,0])
    elif label == 1:
        return np.array([0,1,0,0,0])
    elif label == 2:
        return np.array([0,0,1,0,0])
    elif label == 3:
        return np.array([0,0,0,1,0])
    elif label == 4:
        return np.array([0,0,0,0,1])

def one_hot_label(label):
    if label==0:
        return np.array([1,0])
    else:
        return np.array([0,1])


def review_to_w2v_vector(rev):
    split_words = rev.split()
    
    words_in_order = np.random.randint(len(final_vocab), size=WORDS_TO_TAKE, dtype='int32')
    indexCounter = 0
    good_words = [w for w in split_words if w in final_vocab_lookup]
    
    for word in good_words:
        words_in_order[indexCounter] = final_vocab_lookup[word]
        indexCounter += 1
        if indexCounter >= WORDS_TO_TAKE:
            break
    return words_in_order



def get_amazon_embeds():
    amazon_dir = "./data/amazon/cleaned/"
    cleaned_files = [amazon_dir+f for f in os.listdir(amazon_dir) if f.endswith(".csv")]
    frames = []
    for f in cleaned_files:
        print(f)
        frame = pd.read_csv(f, encoding='utf-8')
        print("frame shape", frame.shape, frame.columns)
        
        frame = frame[(frame.clean_text.notnull()) & (frame.clean_text.str.len() > 100)]
        print("frame shape", frame.shape)
        frames.append(frame)
        
    data = pd.concat(frames)
    print("final shape", data.shape)
    numClasses = 2
    data = sklearn.utils.shuffle(data, random_state=1)
    if data.shape[0] > MAX_DATA_SIZE:
        data = data[0:MAX_DATA_SIZE]
    start_index = 0
    SPLIT_SIZE = 16384
    shuffled_frames = []
    while start_index+SPLIT_SIZE < data.shape[0]:
        shuffled_frames.append(data[start_index:start_index+SPLIT_SIZE])
        start_index += SPLIT_SIZE

    for i in range(int(data.shape[0] / SPLIT_SIZE)):

        file_name = "domain_w2vreviews_" + str(i)
        label_file_name = "domain_w2vlabels_" + str(i)
        stars_file_name = "domain_w2vstars_" + str(i)
        batch = shuffled_frames[i]
        
        
        print("batch selected",i, batch.shape)
        vecs = batch.clean_text.map(review_to_w2v_vector)


        arr = np.zeros((SPLIT_SIZE, WORDS_TO_TAKE))
        labels = np.zeros((SPLIT_SIZE, numClasses))

        for j, v in enumerate(vecs):
            arr[j,] = v
            labels[j,] = one_hot_label(batch.sentiment.iloc[j])
            
            
        if not i % 6 == 0:
            # dont oversample, test set
            sm = SMOTE(ratio='minority', random_state=42)
            arr, new_labels = sm.fit_sample(arr, batch.sentiment.values)
            labels = np.zeros((arr.shape[0], numClasses))
            for i, l in enumerate(new_labels):
                labels[i,] = one_hot_label(new_labels[i])
        else:
            first = False
            file_name = file_name.replace("balanced", "test")
            label_file_name = label_file_name.replace("balanced", "test")
        
        print("saving arr", arr.shape)
        np.save("data/amazon/domain_vecs/" + file_name, arr)
        np.save("data/amazon/domain_vecs/" + label_file_name, labels)
    return


def get_yelp_embeds():
    data = pd.read_csv("./data/yelp/cleaned/yelp.csv")
    data = data[data.clean_text.map(lambda x:len(x.split())) > 75]
    data = data[data.sentiment >= 0]
    print("yelp data shape", data.shape)
    print("unique sentiment")
    vecs = data.clean_text.map(review_to_w2v_vector)
    numClasses = len(data.sentiment.unique())
    numCats = 5
    numStars = 5
    
    arr = np.zeros((len(vecs), WORDS_TO_TAKE))
    labels = np.zeros((len(vecs), numClasses))
    stars = np.zeros((len(vecs), numStars))
    cats =  np.zeros((len(vecs), numCats))
    for j, v in enumerate(vecs):
        arr[j,] = v
        labels[j,] = one_hot_label(data.sentiment.iloc[j])
        cats[j,] = one_hot_category(data.cat.iloc[j])
        stars[j,] = one_hot_rating(data.stars.iloc[j])

    np.save("./data/yelp/domain_vecs/domain_reviews", arr)
    np.save("./data/yelp/domain_vecs/donain_labels", labels)
    np.save("./data/yelp/domain_vecs/domain_cats", cats)
    np.save("./data/yelp/domain_vecs/domain_stars", stars)

    print("done saving yelp")
	



if __name__ == '__main__':
    MAX_DATA_SIZE = 5000000
    SPLIT_SIZE = 16834
    WORDS_TO_TAKE = 100
    not_in_vocab = set()
    model = gensim.models.KeyedVectors.load_word2vec_format("w2v/GoogleNews-vectors-negative300.bin", binary=True)

    vocab_name = "domain_vocab"
    final_vocab, final_vocab_lookup = get_vocab(vocab_name)
    
    VOCAB_SIZE = len(final_vocab)
    print("vocab loaded, size", VOCAB_SIZE)

    word_vectors = np.zeros((VOCAB_SIZE, 300))

    for i, v in enumerate(final_vocab):
        word_vectors[i,] = model.wv[v]
        
    np.save("data/domain_w2v_vectors.npy", word_vectors)
    print("saved word vectors")
    get_amazon_embeds()
    get_yelp_embeds()
    

    print("all done")

