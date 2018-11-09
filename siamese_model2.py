import numpy as np
import tensorflow as tf
from omniglot_preprocesser import omniglot_preprocess

W_init = tf.initializers.random_normal(0.0, 0.01)
b_init = tf.initializers.random_normal(0.5, 0.01)
reg_128 = tf.contrib.layers.l2_regularizer(scale=0.01)
reg_256 = tf.contrib.layers.l2_regularizer(scale=0.01)
reg_4096 = tf.contrib.layers.l2_regularizer(scale=0.001)


#define the model
def model(input_tensor,reuse=False):
	#implemented as in paper, but with a Xavier initializer
	with tf.variable_scope("conv_1") as scope:
		net = tf.contrib.layers.conv2d(input_tensor, 64, [10, 10], activation_fn=tf.nn.relu, padding='VALID',
	        weights_initializer=W_init,scope=scope,reuse=reuse)
		net = tf.contrib.layers.max_pool2d(net, [2, 2],stride=2, padding='VALID')

	with tf.variable_scope("conv_2") as scope:
		net = tf.contrib.layers.conv2d(net, 128, [7, 7], activation_fn=tf.nn.relu, padding='VALID',
	        weights_initializer=W_init, biases_initializer = b_init,scope=scope,reuse=reuse)
		net = tf.contrib.layers.max_pool2d(net, [2, 2], stride=2, padding='VALID') 

	with tf.variable_scope("conv_3") as scope:
		net = tf.contrib.layers.conv2d(net, 128, [4, 4], activation_fn=tf.nn.relu, padding='VALID',
	        weights_initializer=W_init, biases_initializer = b_init, weights_regularizer=reg_128,scope=scope,reuse=reuse)
		net = tf.contrib.layers.max_pool2d(net, [2, 2], stride=2, padding='VALID')
	with tf.variable_scope("conv_4") as scope:
		net = tf.contrib.layers.conv2d(net, 256, [4, 4], activation_fn=tf.nn.relu, padding='VALID',
	        weights_initializer=W_init, biases_initializer=b_init, weights_regularizer=reg_256,scope=scope,reuse=reuse)

	net = tf.contrib.layers.flatten(net)
	with tf.variable_scope("fc_2") as scope:
		net = tf.contrib.layers.fully_connected(net, 4096, weights_initializer=W_init, biases_initializer=b_init, weights_regularizer=reg_4096,
			activation_fn=tf.nn.sigmoid,scope=scope,reuse=reuse)

	return net

def L1_norm(left,right):
	return tf.abs(tf.subtract(encoded_left,encoded_right))

def Final_Output(L1_norm):
	return tf.contrib.layers.fully_connected(L1_norm,1,activation_fn=None)


def compute_cost(logits,labels):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels) + tf.losses.get_regularization_loss())


#Create placeholders

left_input =  tf.placeholder(tf.float32,shape=(None,105,105,1))
right_input = tf.placeholder(tf.float32,shape=(None,105,105,1))

Y = tf.placeholder(tf.float32,shape=(None,1))

encoded_left = model(left_input,reuse=False)
encoded_right = model(right_input,reuse=True)

encoded_distance = L1_norm(encoded_left,encoded_right)
#print(encoded_distance.shape)
final_output = Final_Output(encoded_distance)
#print(final_output.shape)
sigmoid_output = tf.sigmoid(final_output)

cost = compute_cost(final_output,Y)

optimizer = tf.train.AdamOptimizer(0.00006)
train_step = optimizer.minimize(cost)

#Load the data

PATH = '/Users/DivyanshuMurli1/Google Drive/Personal_projects/Omniglot'
Siamese_loader = omniglot_preprocess(PATH,['train','val'])
data, classes = Siamese_loader.load_pickled_data

pairs,targets  = Siamese_loader.get_training_batches(data,32)
print(pairs[0].shape)
pairs_val, targets_val = Siamese_loader.create_one_shot_validator(data,20)
print(targets_val.shape)
print(pairs_val[1].shape)


with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())

	for i in range(100):

		pairs,targets  = Siamese_loader.get_training_batches(data,32)
		#print(targets)
		_, cost_ = sess.run([train_step,cost], feed_dict = {left_input: pairs[0], right_input: pairs[1], Y:targets})
		print("Loss: " + str(cost_))
		
		if i % 10 == 0:
			n_correct = 0
			for i in range(25):
				inputs, _ = Siamese_loader.create_one_shot_validator(data, 20)
				#final_prob = sess.run([final_output], feed_dict = {left_input: inputs[0], right_input: inputs[1]})
				#max_prob = sess.run([sigmoid_output], feed_dict = {left_input: inputs[0], right_input: inputs[1]})
				max_prob_pos = sess.run([tf.argmax(sigmoid_output)], feed_dict = {left_input: inputs[0], right_input: inputs[1]})
				#print(max_prob_pos[0][0])
				#print(final_prob)
				#print(max_prob)
				if max_prob_pos[0][0] == 0:
					n_correct += 1
			print("Accuracy on val set: " + str(n_correct))
		"""
		if i % 10 == 0:
			inputs, _ = Siamese_loader.create_one_shot_validator(data, 20)
			print(sess.run([final_output], feed_dict = {left_input: inputs[0], right_input: inputs[1]}))
		"""


#training seems very slow











