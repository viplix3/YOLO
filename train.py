import argparse
import tensorflow as tf
from tqdm import tqdm
import time
import os
from utils.yolo_loss import compute_loss
from utils.read_tfrecord import *
from utils import checkmate
from inference import load_graph
from time import time
from utils.checkmate import BestCheckpointSaver
from converter import convert
import numpy as np

parser = argparse.ArgumentParser(description='Fine-Tuning YOLO on your own dataset')
parser.add_argument('dataset', help='COCO or VOC or other')
parser.add_argument('num_epochs', help='Number of epochs for model training')
parser.add_argument('checkpoint_dir', help='Path for saving the training checkpoints')
parser.add_argument('--class_path', 
	help='if dataset is other than COCO or VOC, path for the dataset classes')
parser.add_argument('--gpu-num', help="GPU to be used for running the inference", 
	default=0)
parser.add_argument('--logs_dir', help='Path for saving the logs', 
	default='./logs')
parser.add_argument('--anchors_path', help='Path having the YOLO anchors.txt file', 
	default='./yolo_anchors.txt')
parser.add_argument('--tfrecords_dir', help='Path having the tfrecord file(s)',
	default='./tfrecords')


def get_classes(classes_path):
	""" Loads the classes 
		Input:
			classes_path: path to the file containing class names
		Output: list containing class names
	"""

	with open(classes_path) as f:
	    class_names = f.readlines()
	class_names = [c.strip() for c in class_names]
	return class_names


def get_tfrecords(tfrecord_dir):
	""" Fethches the tfrecords for training.
		Input:
			tfrecords_dir: directory in which tfrecords are kept
		Output:
			tfrecords_train: tfrecords containing the training data
			tfrecords_val: tfrecords containing the validation data
	"""
	tfrecords = tfrecord_auto_traversal(tfrecord_dir, os.listdir(tfrecord_dir))

	tfrecords.sort()
	tfrecords_train = tfrecords[:-2]
	tfrecords_val = tfrecords[-2:]

	return tfrecords_train, tfrecords_val


def read_anchors(file_path):
	""" Reads the anchors computer by k-means.py for from the provided path
		Input:
			file_path: path to anchors.txt contaning the anchors computer by k-means.py
		Output:
			A numpy array containing the anchors written into anchors.txt
	"""
	anchors = []
	with open(file_path, 'r') as file:
		for line in file.read().splitlines():
			w, h = line.split()
			anchor = [float(w), float(h)]
			anchors.append(anchor)

	return np.asarray(anchors)


def train(num_epochs, ckpt_path, log_path, anchors_path, tfrecord_path, dataset, class_path,
	decay_steps=5, decay_rate=0.99, batch_size=4):
	""" Function to train the model.
		num_epochs: int, number of epochs for training the model
		ckpt_path: path for saving/restoring the model
		log_path: path for saving the training/validation logs
		anchors_path: path for YOLO anchors
		tfrecord_path: path or tfrecords
		dataset: dataset on which the model is being trained on
		decay_steps: int, steps after which the learning rate is to be decayed
		decay_rate: float, rate to carrying out exponential decay
		batch_size: int, size of the batch for training
	"""

	# Getting the anchors
	anchors = read_anchors(anchors_path)
	tfrecords_train, tfrecords_val = get_tfrecords(tfrecord_path)
	if dataset == 'COCO':
		class_path = './model_data/coco_classes.txt'
	elif dataset == 'VOC':
		class_path = './model_data/voc_classes.txt'
	else:
		assert not class_path == None, 'path to classes.txt is required when dataset is other than COCO or VOC'

	classes = get_classes(class_path)
	num_classes = len(classes)
	
	# Loading the pre-defined Graph
	if not os.path.exists(ckpt_path):
		os.mkdir(ckpt_path)
		convert('./darknet_data/yolov3.cfg', './darknet_data/yolov3.weights', ckpt_path, True, num_classes)
	
	else:
		if len(os.listdir(ckpt_path)) == 0:
			convert('./darknet_data/yolov3.cfg', './darknet_data/yolov3.weights', ckpt_path, True, num_classes)

	input_node, output, is_training, sess = load_graph(ckpt_path)

	graph = tf.get_default_graph()

	with graph.as_default():

		# moving_mean = tf.get_default_graph().get_tensor_by_name("convolutional_0/batch_norm/parameters/moving_mean:0")
		# moving_variance = tf.get_default_graph().get_tensor_by_name("convolutional_0/batch_norm/parameters/moving_variance:0")

		# Declaring the parameters for training the model
		with tf.name_scope('train_parameters'):
			epoch_loss = []
			global_step = tf.Variable(0, trainable=False, name='global_step')
			learning_rate = tf.Variable(5*1e-6, trainable=False, name='learning_rate')
			learning_rate = tf.train.exponential_decay(learning_rate, global_step,
				decay_steps, decay_rate)
			tf.summary.scalar('learning rate', learning_rate)

		# Declaring the parameters for GT
		with tf.name_scope('Targets'):
			scale1 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None, 5+num_classes],
				name='scale1_placeholder')
			scale2 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None, 5+num_classes],
				name='scale2_placeholder')
			scale3 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None, 5+num_classes],
				name='scale3_placeholder')
		y_true = [scale1, scale2, scale3]

		x_reshape = tf.reshape(input_node, [-1, 416, 416, 1])
		tf.summary.image('input', x_reshape)

		# Compute Loss
		with tf.name_scope('Loss_and_Detect'):
			loss = compute_loss(output, y_true, anchors, num_classes, print_loss=False)
			tf.summary.scalar('Loss', loss)

		# Define an optimizer for minimizing the computed loss
		with tf.name_scope('Optimizer'):
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,
					global_step=global_step)

		# A saver object for saving the model
		saver = tf.train.Saver()

		# Getting all the summaries
		summary_op = tf.summary.merge_all()

		# Defining the summary writers for training and validation
		train_summary_writer = tf.summary.FileWriter(
			os.path.join(log_path, os.path.join(dataset, 'train')), sess.graph)
		validation_summary_writer = tf.summary.FileWriter(
			os.path.join(log_path, os.path.join(dataset, 'validation')), sess.graph)


		if os.path.exists(os.path.join(ckpt_path, 'checkpoint')):
			if len(os.listdir(ckpt_path)) > 4:
				saver.restore(sess, checkmate.get_best_checkpoint(ckpt_path))
				print('Model Loaded!')
			# # Initializing all the defined tensorflow graph parameters 
			else:
				sess.run(tf.global_variables_initializer())
		else:
			sess.run(tf.global_variables_initializer())

		train_image_tensor, train_label1_tensor, train_label2_tensor, train_label3_tensor = read_tf_records(tfrecords_train,
			batch_size=batch_size) 

		val_image_tensor, val_label1_tensor, val_label2_tensor, val_label3_tensor = read_tf_records(tfrecords_val, 
			batch_size=batch_size) 


		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		best_ckpt_saver = BestCheckpointSaver(save_dir=ckpt_path, num_to_keep=5)

		for epoch in range(int(num_epochs)):
			tick = time()

			mean_loss_train = []

			for k in tqdm(range(180//batch_size)):

				batch_image_train, batch_label1_train, batch_label2_train, batch_label3_train = sess.run([train_image_tensor, 
						train_label1_tensor, train_label2_tensor, train_label3_tensor])

				summary_train, loss_train, _ = sess.run([summary_op, loss,
					optimizer], feed_dict={input_node: batch_image_train/255.,
											is_training: True, 
											scale1: batch_label1_train,
											scale2: batch_label2_train,
											scale3: batch_label3_train})

				train_summary_writer.add_summary(summary_train, epoch)
				train_summary_writer.flush()
				mean_loss_train.append(loss_train)

			mean_loss_train = np.mean(mean_loss_train)


			mean_loss_valid = []
			print('Validating.....')

			for k in range(20//batch_size):

				batch_image_val, batch_label1_val, batch_label2_val, batch_label3_val = sess.run([val_image_tensor, 
						val_label1_tensor, val_label2_tensor, val_label3_tensor])

				summary_val, loss_valid = sess.run([summary_op, loss],
					feed_dict={input_node: batch_image_val/255.,
								is_training: False,
								scale1: batch_label1_val,
								scale2: batch_label2_val,
								scale3: batch_label3_val})

				# print(loss_valid.shape)

				validation_summary_writer.add_summary(summary_val, epoch)
				validation_summary_writer.flush()
				mean_loss_valid.append(loss_valid)

			mean_loss_valid = np.mean(mean_loss_valid)

			print('Epoch: {} completed\nTraining_loss: {}, Validation Loss: {}'.format(
				epoch+1, mean_loss_train, mean_loss_valid))


			# if (epoch+1) % 10 == 0:
			# 	best_ckpt_saver.handle(loss_valid, sess, global_step)
			best_ckpt_saver.handle(mean_loss_valid, sess, global_step)

		print('Tuning Completed!!')
		train_summary_writer.close()
		validation_summary_writer.close()
		sess.close()








def main(args):
	""" main function which calls all the other required functions for training
		Input:
			args : argument parser object containing the required command line arguments
	"""
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
	train(args.num_epochs, args.checkpoint_dir, args.logs_dir, args.anchors_path, 
		args.tfrecords_dir, args.dataset, args.class_path)



if __name__ == '__main__':
	main(parser.parse_args())