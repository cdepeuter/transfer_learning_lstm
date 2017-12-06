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
    review_sizes[len(good_words)] += 1
    
    for word in good_words:
        words_in_order[indexCounter] = final_vocab_lookup[word]
        indexCounter += 1
        if indexCounter >= WORDS_TO_TAKE:
            break
    return words_in_order



def get_amazon_embeds():
    amazon_dir = "./data/amazon/cleaned/"
    cleaned_files = [amazon_dir+f for f in os.listdir(amazon_dir)]

    for f in cleaned_files:
        print(f)
        frame = pd.read_csv("data/"+f, encoding='utf-8')
        print("frame shape", frame.shape, frame.columns)
        
        frame = frame[(frame.clean_text.notnull()) & (frame.clean_text.str.len() > 100)]
        print("frame shape", frame.shape)
        frames.append(frame)
        
    data = pd.concat(frames)

    data = sklearn.utils.shuffle(data, random_state=1)
    # only keep reviews with so many words
    data = data[data.clean_text.map(lambda x:len(x.split())) > 75]
    print("final shape", data.shape)

    start_index = 0
    SPLIT_SIZE = 16384
    shuffled_frames = []
    while start_index+SPLIT_SIZE < data.shape[0]:
        shuffled_frames.append(data[start_index:start_index+SPLIT_SIZE])
        start_index += SPLIT_SIZE

    for i in range(int(data.shape / SPLIT_SIZE)):

        file_name = "balanced_w2vreviews_" + str(i)
        label_file_name = "balanced_w2vlabels_" + str(i)
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
        np.save("data/amazon/vecs/" + file_name, arr)
        np.save("data/amazon/vecs/" + label_file_name, labels)
    return


def get_yelp_embeds():
    data = pd.read_csv("./data/yelp/cleaned/yelp.csv")
    data = data[data.clean_text.map(lambda x:len(x.split())) > 75]
    print("yelp data shape", data.shape)
    
    vecs = data.clean_text.map(review_to_w2v_vector)
    numClasses = len(data.sentiment.unique())
    numCats = len(data.cat.unique())
    
    arr = np.zeros((len(vecs), WORDS_TO_TAKE))
    labels = np.zeros((len(vecs), numClasses))
    cats =  np.zeros((len(vecs), numCcats))

    for j, v in enumerate(vecs):
        arr[j,] = v
        labels[j,] = one_hot_label(data.sentiment.iloc[j])
        cats[j,] = one_hot_category(data.cat.iloc[j])

    np.save("./data/yelp/vecs/reviews", arr)
    np.save("./data/yelp/vecs/labels", labels)
    np.save("./data/yelp/vecs/cats", cats)

    print("done saving yelp")
	

def is_valid_word(w):
    """
        Only want lower case words that aren't stopwords and arent tags or other nonsense like ####
    """
    if not w.lower() == w:
        return False
#     if w in stop_words:
#         return False
    if re.search("[^(\w|\'|\-)]", w):
        return False
#     if not w in model.wv.vocab:
#         # shouldnt have to add this but for some reason it makes a difference, where are these words coming from?
#         return False
    
    return True


if __name__ == '__main__':
    SPLIT_SIZE = 16834
    WORDS_TO_TAKE = 100
    not_in_vocab = set()
    VOCAB_SIZE = 400000
	
    model = gensim.models.KeyedVectors.load_word2vec_format("w2v/GoogleNews-vectors-negative300.bin", binary=True)
    vocab_counts = [(word, vocab_obj.count) for  (word, vocab_obj) in model.vocab.items() if is_valid_word(word)]
    vocab_counts = sorted(vocab_counts, key=lambda x:x[1], reverse=True)
    print("found counts")
    # needs a list for ordering
    final_vocab = [v[0] for v in vocab_counts[0:VOCAB_SIZE]]
    print("vocab set")
    # get a lookup for O[1] access
    final_vocab_lookup = {v:final_vocab.index(v) for v in final_vocab}
    reverse_lookup = {v: k for k, v in final_vocab_lookup.items()}

    word_vectors = np.zeros((VOCAB_SIZE, 300))

    for i, v in enumerate(final_vocab):
        if v not in model.wv.vocab:
            print("huh", v)
        word_vectors[i,] = model.wv[v]
        
    np.save("data/w2v_vectors.npy", word_vectors)
    print("saved word vectors")

    get_yelp_embeds()
    get_amazon_embeds()

    print("all done")

