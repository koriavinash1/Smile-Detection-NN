# -*- coding: utf-8 -*-
import cv2, os, random
import tensorflow as tf
import numpy as np
from PIL import Image

sess = tf.InteractiveSession()

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
timageh,timagew = 100,100#test image height and width(h,w)....
path = './yalefaces'
images={'image':[],'label':[]}

def get_image_data():
	global images
	image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
	for image_path in image_paths:
		image_pil = Image.open(image_path).convert('L')
		image = np.array(image_pil, 'uint8')
		nbr = os.path.split(image_path)[1].split(".")[1]
		faces = faceCascade.detectMultiScale(image)
		for (x, y, w, h) in faces:
			if nbr == "happy":
				images['image'].append(tf.reshape(cv2.resize(image[y:y+h, x:x+w],(timagew, timageh), interpolation = cv2.INTER_CUBIC),[-1]))
				images['label'].append([1,0,0,0,0,0,0])
			elif nbr == "sad":
				images['image'].append(tf.reshape(cv2.resize(image[y:y+h, x:x+w],(timagew, timageh), interpolation = cv2.INTER_CUBIC),[-1]))
				images['label'].append([0,1,0,0,0,0,0])
			elif nbr == "glasses":
				images['image'].append(tf.reshape(cv2.resize(image[y:y+h, x:x+w],(timagew, timageh), interpolation = cv2.INTER_CUBIC),[-1]))
				images['label'].append([0,0,1,0,0,0,0])
			elif nbr == "surprised":
				images['image'].append(tf.reshape(cv2.resize(image[y:y+h, x:x+w],(timagew, timageh), interpolation = cv2.INTER_CUBIC),[-1]))
				images['label'].append([0,0,0,1,0,0,0])
			elif nbr == "sleepy":
				images['image'].append(tf.reshape(cv2.resize(image[y:y+h, x:x+w],(timagew, timageh), interpolation = cv2.INTER_CUBIC),[-1]))
				images['label'].append([0,0,0,0,1,0,0])
			elif nbr == "wink":
				images['image'].append(tf.reshape(cv2.resize(image[y:y+h, x:x+w],(timagew, timageh), interpolation = cv2.INTER_CUBIC),[-1]))
				images['label'].append([0,0,0,0,0,1,0])
			else:
				images['image'].append(tf.reshape(cv2.resize(image[y:y+h, x:x+w],(timagew, timageh), interpolation = cv2.INTER_CUBIC),[-1]))
				images['label'].append([0,0,0,0,0,0,1])
			cv2.imshow("Adding faces to traning set...", cv2.resize(image[y:y+h, x:x+w],(5*timagew, 5*timageh), interpolation = cv2.INTER_CUBIC))
			cv2.waitKey(5)
	print images


def fully_connected_net_no_hidden_layers():
	x = tf.placeholder(tf.float32, shape=[None, timageh*timagew])
	y_ = tf.placeholder(tf.float32, shape=[None, 7])
	W = tf.Variable(tf.zeros([timagew*timageh,7]))
	b = tf.Variable(tf.zeros([7]))
	sess.run(tf.global_variables_initializer())
	y = tf.matmul(x,W) + b
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	for _ in range(400):
		train_step.run(feed_dict={x: sess.run(images['image']), y_: images['label']})
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print "accuracy................", accuracy.eval(feed_dict={x: sess.run(images['image']), y_: images['label']})

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def Convolutional_net():
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	x_image = tf.reshape(sess.run(images['image'][0]), [-1,timageh,timagew,1])
	h_conv1 = tf.nn.relu(conv2d(images['image'], W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
	cv2.imshow("temp", sess.run(h_pool1))
	cv2.waitKey(0)
	pass

get_image_data()
fully_connected_net_no_hidden_layers()
# Convolutional_net()