import tensorflow as tf
import numpy as np
import os
import re
import datetime
import codecs
import sys


"""
LSTM for Yelp,

inputs for what target to predict:

    'semntiment' : target is [0,1] sentiment
    'rating' : [1-5] stars
    'category': [0-4] category

$ python yelp_lstm category 512 32 1000

"""
current_time =  datetime.datetime.now().strftime("%Y%m%d-%H%M")
TRAIN_SIZE=None
iterations = 100
maxSeqLength = 100
numDimensions = 300
numClasses = 2
BATCH_SIZE = 512
lstmUnits = 16
target = 'sentiment'

if len(sys.argv) > 1:
    target = sys.argv[1]
    if len(sys.argv) > 2:
        TRAIN_SIZE = int(sys.argv[2])
        if len(sys.argv) > 3:
            BATCH_SIZE = int(sys.argv[3])
            lstmUnits = int(sys.argv[4])
            iterations = int(sys.argv[5])

print("batch size", BATCH_SIZE)
print("lstmUnits", lstmUnits)
print("iterations", iterations)

str_params = [current_time, target, str(BATCH_SIZE), str(lstmUnits), str(iterations), str(TRAIN_SIZE)]

VECTORS_FILE = "data/w2v_vectors.npy"
DATA_FILE_START = "yelp_w2vreviews"

wordVectors = np.load(VECTORS_FILE)
print("wordVectors shape", wordVectors.shape)


def getYelpData():
    X = np.load("data/yelp/vecs/reviews.npy")
    if target == 'sentiment':  
        y = np.load("data/yelp/vecs/labels.npy")
    elif target == "cats":
        y = np.load("data/yelp/vecs/cats.npy")
    elif target == "stars":
        y = np.load("data/yelp/vecs/stars.npy")

    # shuffle the data
    shuffle = np.random.permutation(np.arange(X.shape[0]))
    X = X[shuffle,]
    y = y[shuffle,]
    if TRAIN_SIZE is not None:
        X = X[:TRAIN_SIZE,]
        y = y[:TRAIN_SIZE,]

    # split into train and test sets
    split_at = int(4*X.shape[0]/5)
    X_train = X[0:split_at,]
    y_train = y[0:split_at,]
    X_test = X[split_at:,]
    y_test = y[split_at:,]

    return X_train.astype(int),y_train.astype(int), X_test.astype(int), y_test.astype(int)



def getTrainBatch(size):
    global train_data
    global train_labels
    ix = np.random.randint(train_data.shape[0], size=size)
    return train_data[ix,], train_labels[ix,]


train_data, train_labels, test_data, test_labels = getYelpData()

print("train data shape", train_data.shape)
print("train labels shape", train_labels.shape)
print("train data max:", train_data.max())
print("test labels and data shapes",test_labels.shape, test_data.shape)
print("test data balance", test_labels.mean(axis=0))
print("train data balance", train_labels.mean(axis=0))

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [None, numClasses])
input_data = tf.placeholder(tf.int64, [None, maxSeqLength])

keep_prob = tf.placeholder(tf.float32)
embedding = tf.get_variable(name="word_embedding", shape=wordVectors.shape, initializer=tf.constant_initializer(wordVectors), trainable=False)


#data = tf.Variable(tf.zeros([None, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(embedding,input_data)
#print("data", data[1,])

# lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits,forget_bias=1)
# outputs,_ =tf.contrib.rnn.static_rnn(lstmCell,input,dtype="float32")

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmDropout = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=keep_prob)
#outputs, states = tf.nn.dxqynamic_rnn(lstmCell, data, dtype=tf.float32)

value, _ = tf.nn.dynamic_rnn(lstmDropout, data, dtype=tf.float32)


weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]),  name="last_layer_weights")
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]), name="last_layer_biases")
value = tf.transpose(value,   [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)


correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(loss)


sess = tf.InteractiveSession()
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + current_time + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
final_yelp_acc = None
final_test_acc = None


# load_path = "./models/final_lstm20171205-1304.ckpt"

# # Initialize v1 since the saver will not.
# saver.restore(sess, load_path)


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
        print("****************")
        print("test accuracy:  % f" % final_test_acc)
        print("\n\n")

if TRAIN_SIZE is None:
    TRAIN_SIZE = test_data.shape[0] + train_data.shape[0]
with codecs.open("logs/yelp_final_"+'_'.join(str_params)+".txt",'w') as fp:
    fp.write("time " + current_time + "\n")
    fp.write("target" + target)
    fp.write("BATCH_SIZE " + str(BATCH_SIZE) + "\n")
    fp.write("TRAIN SIZE " + str(TRAIN_SIZE) + "\n")
    fp.write("lstmUnits " + str(lstmUnits) + "\n")
    fp.write("iterations " + str(iterations) + "\n")
    fp.write("test accuracy " + str(final_test_acc) + "\n")
    # fp.write("yelp accuracy " + str(final_yelp_acc) + "\n")


# with codecs.open("logs/run_final_"+current_time+".txt",'w') as fp:
#     fp.write("time " + current_time + "\n")
#     fp.write("BATCH_SIZE " + str(BATCH_SIZE) + "\n")
#     fp.write("TEST SIZE " + str(TEST_SIZE) + "\n")
#     fp.write("lstmUnits " + str(lstmUnits) + "\n")
#     fp.write("iterations " + str(iterations) + "\n")
#     fp.write("test accuracy " + str(final_test_acc) + "\n")
#     fp.write("yelp accuracy " + str(final_yelp_acc) + "\n")

