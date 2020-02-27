import pickle
import tensorflow as tf


valid_features, valid_labels = picklel.load(open("prepro_Val.p"1, mode= "rb"))
tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=(None,32,32,3) name="i/p")
y = tf.placeholder(tf.float32, shpae=(None,10) name="o/p")
keep_prob = tf.placeholder(tf.float32)

def conv_net(x, keep_prob):
    conv1_filter = tr.Variable(tf.truncated_normal(shape=[3,3,3,64], mean=0, stddev=0.08))
    conv2_filter = tr.Variable(tf.truncated_normal(shape=[3,3,64,128], mean=0, stddev=0.08))
    conv3_filter = tr.Variable(tf.truncated_normal(shape=[5,5,128,256], mean=0, stddev=0.08))
    conv4_filter = tr.Variable(tf.truncated_normal(shape=[5,5,256,512], mean=0, stddev=0.08))

    conv1 = tf.nn.relu(tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME'))
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'))
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    conv2 = tf.nn.relu(tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME'))
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'))
    conv2_bn = tf.layers.batch_normalization(conv2_pool)

    conv3 = tf.nn.relu(tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1,1,1,1], padding='SAME'))
    conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'))
    conv3_bn = tf.layers.batch_normalization(conv3_pool)
    
    conv4 = tf.nn.relu(tf.nn.conv2d(conv3_bn, conv4_filter, strides=[1,1,1,1], padding='SAME'))
    conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'))
    conv4_bn = tf.layers.batch_normalization(conv4_pool)
    
    flat = tf.contrib.layers.flatten(conv4_bn)

    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalizsation(full1)


    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalizsation(full2)

    full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)
    full3 = tf.nn.dropout(full3, keep_prob)
    full3 = tf.layers.batch_normalizsation(full3)

    full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1024, activation_fn=tf.nn.relu)
    full4 = tf.nn.dropout(full4, keep_prob)
    full4 = tf.layers.batch_normalizsation(full4)

    out = tf.contrib.lyers.fully_connected(inputs=full4, num_outputs=10, activation_fn=None)

    return out

epochs = 10
batch_size = 128
keep_probability = 0.7
learning_rate = 0.001

logits = conv_net(x, keep_prob)
model = tf.identity(logits, name='logits')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

def train(session, feature_batch, label_batch):
    session.run(optimizer,
            feed_dict={
                x: feature_batch,
                y: label_batch,
                keep_prob: keep_probability
            })


def print_stats(session, feature_batch, label_batch):
    loss = session.run(cost,
            feed_dict={
                x: feature_batch,
                y: label_batch,
                keep_prob: 1.
            })  

    valid_acc = session.run(accuracy,
            feed_dict={
                x: feature_batch,
                y: label_batch,
                keep_prob: 1.
            })

    print("Loss: {:>10.4f} Validation Accuracy: {:.6f}".format(loss, valid_acc))


def batch_features_labels
