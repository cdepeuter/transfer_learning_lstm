import tensorflow as tf
import numpy as np
import os
import re
import datetime
import codecs
import sys


"""
Regular LSTM for training/predicting on Amazon review sentiment. 
Load all data from data/amazon/vecs (test files have a different file prefix)

Also load Yelp sentiment data and see how the trained dataset does on predicting
Yelp sentiment. But never train on this yelp data

"""

current_time =  datetime.datetime.now().strftime("%Y%m%d-%H%M")
TEST_SIZE= 2084
iterations = 100
maxSeqLength = 100
numDimensions = 300
numClasses = 2
BATCH_SIZE = 512
lstmUnits = 48
data_vecs_path = "./data/amazon/vecs/"
VECTORS_FILE = "data/domain_w2v_vectors.npy"
DATA_FILE_START = "balanced_w2vreviews"
TEST_FILE_START = "test_"
if len(sys.argv) > 1:
    BATCH_SIZE = int(sys.argv[1])
    lstmUnits = int(sys.argv[2])
    iterations = int(sys.argv[3])

print("batch size", BATCH_SIZE)
print("lstmUnits", lstmUnits)
print("iterations", iterations)

# load numpy models
wordVectors = np.load(VECTORS_FILE)
print("wordVectors shape", wordVectors.shape)

str_params = [current_time, str(BATCH_SIZE), str(lstmUnits), str(iterations)]
with codecs.open("logs/amazon_lstm_"+'_'.join(str_params)+".txt",'w') as fp:
    fp.write("time " + current_time + "\n")
    fp.write("BATCH_SIZE " + str(BATCH_SIZE) + "\n")
    fp.write("TEST SIZE " + str(TEST_SIZE) + "\n")
    fp.write("lstmUnits " + str(lstmUnits) + "\n")
    fp.write("iterations " + str(iterations) + "\n")




def getAmazonData(max_size=None, training=True):
    
    # get test set of files
    if training:
        files = [f for i, f in enumerate(os.listdir(data_vecs_path))if "reviews" in f and f.endswith(".npy") and i % 5 != 0]
    else:
        files = [f for i, f in enumerate(os.listdir(data_vecs_path)) if "reviews" in f and f.endswith(".npy") and i % 5 == 0]

    print("getting amazon data", len(files))

    frames = [np.load(data_vecs_path + f) for f in files]
    labels = [np.load(data_vecs_path + f.replace("reviews", "labels")) for f in files]
    
    X = np.vstack(frames)
    y = np.vstack(labels)

    if max_size is not None:
        ix = np.random.randint(X.shape[0], size=max_size)
        X = X[ix,]
        y = y[ix,]
    print("full data shape", X.shape[0])
    return X.astype(int),y.astype(int)


def getTrainBatch(size=None):
    global train_data
    global train_labels
        
  
    if size is not None:
        #ix = np.array(range(size))
        ix = np.random.randint(train_data.shape[0], size=size)
    
    return train_data[ix,], train_labels[ix, ]

def getYelpData(size=25000):
    #train_files = [f for f in os.listdir("data/vecs") if f.startswith(DATA_FILE_START) and f.endswith(".npy")]
    
    X = np.load("data/yelp/vecs/domain_reviews.npy")
    y = np.load("data/yelp/vecs/domain_labels.npy")
    ix = np.random.randint(X.shape[0], size=size)
    X = X[ix,]
    y = y[ix,]
    return X.astype(int),y.astype(int)

def one_hot_label(label):
    if label==0:
        return np.array([1,0])
    else:
        return np.array([0,1])

MAX_TRAIN_SIZE = None
# # REMOVE WHEN ON GOOGLE CLOUD_PLATFORM
# MAX_TRAIN_SIZE = 2000000
train_data, train_labels = getAmazonData(max_size=MAX_TRAIN_SIZE)

print("train data shape", train_data.shape)
print("train labels shape", train_labels.shape)
print("train data max:", train_data.max())
print("train data balance", train_labels.mean(axis=0))


yelp_data, yelp_labels = getYelpData()

print("yelp data shape", yelp_data.shape)

test_data, test_labels = getAmazonData(max_size=TEST_SIZE, training=False)

print(len(test_labels), test_labels.shape)
print(test_data.shape)
print("test data balance", test_labels.mean(axis=0))


labels = tf.placeholder(tf.float32, [None, numClasses])
input_data = tf.placeholder(tf.int64, [None, maxSeqLength])
keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope("embeds"):
    embedding = tf.get_variable(name="word_embedding", shape=wordVectors.shape, initializer=tf.constant_initializer(wordVectors), trainable=False)


#data = tf.Variable(tf.zeros([None, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(embedding,input_data)
#print("data", data[1,])

# lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits,forget_bias=1)
# outputs,_ =tf.contrib.rnn.static_rnn(lstmCell,input,dtype="float32")

with tf.variable_scope("lstm"):

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmDropout = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=keep_prob)
    #outputs, states = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    value, _ = tf.nn.dynamic_rnn(lstmDropout, data, dtype=tf.float32)

with tf.variable_scope("final_layer"):

    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]), name="last_layer_weights")
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]), name="last_layer_biases")
    value = tf.transpose(value,   [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)


correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(loss)

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()


sess = tf.InteractiveSession()
logdir = "tensorboard/" + current_time + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

# dont save last layer weights
all_vars = tf.all_variables()
var_to_restore = [v for v in all_vars if v.name.split('/')[0]=='embed' or v.name.split('/')[0]=='lstm']
var_names = [v.name for v in var_to_restore]
saver = tf.train.Saver(var_to_restore)


print("variables to restore", var_names)

sess.run(tf.global_variables_initializer())

final_yelp_acc = None
final_test_acc = None



for i in range(iterations):
    #Next Batch of reviews
    nextBatch, nextBatchLabels = getTrainBatch(size=BATCH_SIZE)
    _, summary, acc, pred, loss_, outputs = sess.run([ train_step, merged, accuracy, prediction, loss, value], {input_data: nextBatch, labels: nextBatchLabels,keep_prob:.75})
    #Save the network every 10,000 training iterations
    if i % 10 == 0:
        
        writer.add_summary(summary, i)
        print("\n")
        print("step %d" % i)
        print("train accuracy %f" % acc)
        print("loss: %f" % loss_)
        print("balance %f, %f" % (nextBatchLabels.mean(axis=0)[0], nextBatchLabels.mean(axis=0)[0] + acc))
        #print("pred mean %f" % np.mean(pred))

    if i % 50 == 0 or i == iterations-1:
        final_test_acc = accuracy.eval({input_data:test_data, labels: test_labels, keep_prob:1.0})
        final_yelp_acc = accuracy.eval({input_data:yelp_data, labels: yelp_labels, keep_prob:1.0})
        print("****************")
        print("test accuracy:  % f" % final_test_acc)
        print("yelp accuracy:  % f" % final_yelp_acc)
        print("\n\n")
#         print("mean prediction", np.mean(pred))
#         print("\n\n")
        


with codecs.open("logs/final_amazon_lstm_"+'_'.join(str_params)+".txt",'w') as fp:
    fp.write("time " + current_time + "\n")
    fp.write("BATCH_SIZE " + str(BATCH_SIZE) + "\n")
    fp.write("TEST SIZE " + str(TEST_SIZE) + "\n")
    fp.write("lstmUnits " + str(lstmUnits) + "\n")
    fp.write("iterations " + str(iterations) + "\n")
    fp.write("test accuracy " + str(final_test_acc) + "\n")
    fp.write("yelp accuracy " + str(final_yelp_acc) + "\n")


save_file_path = "./models/final_lstm"+current_time+".ckpt"
save_path = saver.save(sess, save_file_path)
print("saved to %s" % save_file_path)
writer.close()

