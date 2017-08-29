from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# this is a tensor (multidimensional array) of n 784 element 1 hot vectors that represent 28x28 pixel images)
x = tf.placeholder(tf.float32, [None, 784])

# model parameters
# first one is Weight, used for evidence for each of the ten "classes" or possible images
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Model implementation, taking into consideration earlier parameters.
y = tf.nn.softmax(tf.matmul(x, W) + b)

# this is our prediction of how efficient the model will be vs reality - cross-entropy
y_ = tf.placeholder(tf.float32, [None, 10])

# our cross-entropy function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# minimize cross entropy using gradient descent, with a learning rate of .5
# Gradient descent is a simple procedure, where TensorFlow simply shifts each variable a little bit in the direction
# that reduces the cost.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# Well, first let's figure out where we predicted the correct label. tf.argmax is an extremely useful function which
# gives you the index of the highest entry in a tensor along some axis. For example, tf.argmax(y,1) is the label our
# model thinks is most likely for each input, while tf.argmax(y_,1) is the correct label. We can use tf.equal to check
# if our prediction matches the truth.

#determine if model's prediction matches the truth
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#tells us accuracy of test data
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
