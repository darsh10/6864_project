import tensorflow as tf
import numpy as np

tf.reset_default_graph()
inputs = tf.placeholder(shape=[1,2],dtype=tf.float32)
W1 = tf.Variable(tf.random_uniform([2,16],0,0.01))
hidden = tf.nn.sigmoid(tf.matmul(inputs,W1))
W2 = tf.Variable(tf.random_uniform([16,2],0,0.01))
output = tf.nn.softmax(tf.matmul(hidden,W2))
output = tf.reshape(output,[2])
actual = tf.placeholder(shape=[1,2],dtype=tf.float32)
reward = tf.placeholder(shape=[1,2],dtype=tf.float32)
index = tf.placeholder(shape=[1],dtype=tf.int32)
responsible_weight = tf.slice(output,index,[1])
Loss = -tf.log(responsible_weight)
loss = tf.reduce_mean(tf.square(output-actual))
trainer = tf.train.GradientDescentOptimizer(learning_rate=.5)
updateModel = trainer.minimize(Loss)

init = tf.initialize_all_variables()

X = [[0,0],[0,1],[1,0],[1,1]]
Y = [[0,1],[0,1],[0,1],[1,0]]

with tf.Session() as sess:
    sess.run(init)
    oldW = None
    for i in range(1000):
        print
        print
        for j,x in enumerate(X):
            x = np.array(x).reshape(1,2)
            y = np.array(Y[j]).reshape(1,2)
            ind = 0
            if y[0][0] == 0:
                ind = 0
            else:
                ind = 1
            out = sess.run(output,feed_dict={inputs:x})
            _,W = sess.run([updateModel,W1],feed_dict={inputs:x,index:[ind]})
            print x,out
            oldW = W 
