import json
import gensim
import codecs
import re

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
    if REMOVE_DOMAIN_INDEPENDENT_WORDS and w in domain_independent_words:
        return False
    
    return True



domain_independent_words = []
with codecs.open("data/vocab/domain_independent_sentiment.txt", encoding='utf-8') as fp:
    for l in fp:
        domain_independent_words.append(l.strip("\n").strip())

print("len domain indeoendent words", len(domain_independent_words))
VOCAB_SIZE = 400000
REMOVE_DOMAIN_INDEPENDENT_WORDS = True
model = gensim.models.KeyedVectors.load_word2vec_format("w2v/GoogleNews-vectors-negative300.bin", binary=True)
vocab_counts = [(word, vocab_obj.count) for  (word, vocab_obj) in model.vocab.items() if is_valid_word(word)]
vocab_counts = sorted(vocab_counts, key=lambda x:x[1], reverse=True)
print("found counts")
# needs a list for ordering
final_vocab = [v[0] for v in vocab_counts[0:VOCAB_SIZE]]
print("vocab set")
# get a lookup for O[1] access
final_vocab_lookup = {v:final_vocab.index(v) for v in final_vocab}
vocab_name = "domain_vocab"
with codecs.open("data/vocab/"+vocab_name+".json", 'w', encoding='utf-8') as fp:
	fp.write(json.dumps(final_vocab))

with codecs.open("data/vocab/"+vocab_name+"_lookup.json", 'w', encoding='utf-8') as fp:
	fp.write(json.dumps(final_vocab_lookup))

print("done writing vocabs")