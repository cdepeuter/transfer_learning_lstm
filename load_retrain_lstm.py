import tensorflow as tf
import numpy as np
import os
import re
import datetime
import codecs
import sys
from sklearn.model_selection import train_test_split

"""
Retrain the last layer of an lstm

inputs for what target to retrian t0:

    'semntiment' : target is [0,1] sentiment
    'rating' : [1-5] stars
    'category': [0-4] category

$ python load_retrin_lstm category retrin_path batch_size, iterations

"""
current_time =  datetime.datetime.now().strftime("%Y%m%d-%H%M")
iterations = 100
numDimensions = 300
BATCH_SIZE = 2048
TEST_SIZE = 2048
YELP_SIZE = 5000
lstmUnits = 96
target = 'sentiment'
load_path = "./models/final_lstm20171213-1242.ckpt"
#load_path = "./models/final_lstm20171212-1542.ckpt"


if len(sys.argv) > 1:
    target = sys.argv[1]
    YELP_SIZE = int(sys.argv[2])
    if len(sys.argv) > 3:
        load_path = sys.argv[3]
    if len(sys.argv) > 4:
        BATCH_SIZE = int(sys.argv[5])
        iterations = int(sys.argv[6])


print("batch size", BATCH_SIZE)
print("lstmUnits", lstmUnits)
print("iterations", iterations)

str_params = [current_time, target, str(YELP_SIZE), str(BATCH_SIZE), str(iterations), load_path.split("/")[-1].replace(".ckpt", "")]


def getTrainBatch(size):
    global train_data
    global train_labels
    ix = np.random.randint(train_data.shape[0], size=size)
    return train_data[ix,], train_labels[ix,]


def getYelpData(size=None):
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

    if size is not None:
        X = X[0:size,]
        y = y[0:size,]

    # split into train and test sets
    split_at = int(4*X.shape[0]/5)
    X_train = X[0:split_at,]
    y_train = y[0:split_at,]
    X_test = X[split_at:,]
    y_test = y[split_at:,]

    return X_train.astype(int),y_train.astype(int), X_test.astype(int), y_test.astype(int)


#./models/final_lstm20171213-1242.ckpt


VECTORS_FILE = "data/w2v_vectors.npy"
DATA_FILE_START = "yelp_w2vreviews"

wordVectors = np.load(VECTORS_FILE)
print("wordVectors shape", wordVectors.shape)


train_data, train_labels, test_data, test_labels = getYelpData(size=YELP_SIZE)

print("train data shape", train_data.shape)
print("train labels shape", train_labels.shape)
print("train data max:", train_data.max())
print("test labels and data shapes",test_labels.shape, test_data.shape)
print("test data balance", test_labels.mean(axis=0))
print("train data balance", train_labels.mean(axis=0))

maxSeqLength = train_data.shape[1]
numClasses = train_labels.shape[1]

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
with tf.variable_scope("lstm"):
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmDropout = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=keep_prob)
    #outputs, states = tf.nn.dxqynamic_rnn(lstmCell, data, dtype=tf.float32)

    value, _ = tf.nn.dynamic_rnn(lstmDropout, data, dtype=tf.float32)

with tf.variable_scope("final_layer"):
    # old weights will load, but we dont care about that
    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value,   [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)


correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

all_vars = tf.all_variables()
print("all variables", [v.name for v in all_vars])

var_names = [v for v in all_vars if v.name.split("/")[0] == 'final_layer']
print("variables to train")
print(var_names)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(loss, var_list=var_names)


print("NAMES", var_names)

restore_vars = [v for v in all_vars if v.name.split("/")[0] == "lstm"]
sess = tf.InteractiveSession()
saver = tf.train.Saver(var_list=restore_vars)
# Initialize v1 since the saver will not.
saver.restore(sess, load_path)


tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + current_time + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

sess.run(tf.global_variables_initializer())
final_yelp_acc = None
final_test_acc = None




final_test_acc = accuracy.eval({input_data:test_data, labels: test_labels, keep_prob:1.0})
print("****************")
print("test accuracy:  % f" % final_test_acc)

for i in range(iterations):
    #Next Batch of reviews
    nextBatch, nextBatchLabels = getTrainBatch(size=BATCH_SIZE)
    _, summary, acc, loss_,  = sess.run([ train_step, merged, accuracy, loss], {input_data: nextBatch, labels: nextBatchLabels,keep_prob:.75})
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
#         print("mean prediction", np.mean(pred))
#         print("\n\n")
        


str_params = [current_time, target, str(YELP_SIZE), str(BATCH_SIZE), str(iterations), load_path.split("/")[-1].replace(".ckpt", "")]
with codecs.open("logs/retrained_"+'_'.join(str_params)+".txt",'w') as fp:
    fp.write("time " + current_time + "\n")
    fp.write("YELP SIZE " + str(YELP_SIZE) + "\n")
    fp.write("BATCH_SIZE " + str(BATCH_SIZE) + "\n")
    fp.write("load model " + load_path + "\n")
    fp.write("iterations " + str(iterations) + "\n")
    fp.write("target " + target + "\n")
    fp.write("best amazon model " + load_path)
        


# with codecs.open("logs/run_final_"+current_time+".txt",'w') as fp:
#     fp.write("time " + current_time + "\n")
#     fp.write("BATCH_SIZE " + str(BATCH_SIZE) + "\n")
#     fp.write("TEST SIZE " + str(TEST_SIZE) + "\n")
#     fp.write("lstmUnits " + str(lstmUnits) + "\n")
#     fp.write("iterations " + str(iterations) + "\n")
#     fp.write("test accuracy " + str(final_test_acc) + "\n")
#     fp.write("yelp accuracy " + str(final_yelp_acc) + "\n")

