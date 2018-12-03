import argparse
import tensorflow as tf
from tqdm import tqdm
import time
import os
from utils.yolo_loss import compute_loss
from utils.utils import draw_box
from dataParser import Parser
from utils import checkmate
from inference import load_graph
from time import time
from utils.checkmate import BestCheckpointSaver
from converter import convert
import numpy as np
import config

parser = argparse.ArgumentParser(description='Fine-Tuning YOLO on your own dataset')
parser.add_argument('dataset', help='COCO or VOC or other')
parser.add_argument('checkpoint_dir', help='Path for saving the training checkpoints')
parser.add_argument('--class_path', 
	help='if dataset is other than COCO or VOC, path for the dataset classes')
parser.add_argument('--gpu-num', help="GPU to be used for running the inference", 
	default=0)
parser.add_argument('--logs_dir', help='Path for saving the logs', 
	default='./logs')
# parser.add_argument('--anchors_path', help='Path having the YOLO anchors.txt file', 
# 	default='./yolo_anchors.txt')
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



def list_tfrecord_file(folder, file_list):

    
	tfrecord_list = []
	for i in range(len(file_list)):
		current_file_abs_path = os.path.join(folder, file_list[i])		
		if current_file_abs_path.endswith(".tfrecord"):
			tfrecord_list.append(current_file_abs_path)
			#print("Found %s successfully!" % file_list[i])				
		else:
			pass
	return tfrecord_list



# Traverse current directory
def tfrecord_auto_traversal(folder, current_folder_filename_list):

	if current_folder_filename_list != None:
		print("%s files were found under %s folder. " % (len(current_folder_filename_list), 
			folder))
		print("Please be noted that only files ending with '*.tfrecord' will be loaded!")
		tfrecord_list = list_tfrecord_file(folder, current_folder_filename_list)
		if len(tfrecord_list) != 0:
			print("Found %d files:\n %s\n\n\n" %(len(tfrecord_list), 
				current_folder_filename_list))
		else:
			print("Cannot find any tfrecord files, please check the path.")
	return tfrecord_list


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


def train(ckpt_path, log_path, tfrecord_path, dataset, class_path=None,
	decay_steps=2000, decay_rate=0.8):
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
	anchors = read_anchors(config.anchors_path)
	tfrecords_train, tfrecords_val = get_tfrecords(tfrecord_path)
	if dataset == 'COCO':
		class_path = './model_data/coco_classes.txt'
	elif dataset == 'VOC':
		class_path = './model_data/voc_classes.txt'
	elif dataset == 'raccoon':
		class_path = './model_data/raccoon_classes.txt'
	else:
		assert not class_path == None, 'path to classes.txt is required when dataset is other than COCO or VOC'

	classes = get_classes(class_path)
	num_classes = len(classes)

	train_reader = Parser('train', config.data_dir, config.anchors_path, config.output_dir, 
		config.num_classes, input_shape=config.input_shape, max_boxes=config.max_boxes)
	train_data = train_reader.build_dataset(config.train_batch_size)
	iterator = train_data.make_one_shot_iterator()
	images, bbox, bbox_true_13, bbox_true_26, bbox_true_52 = iterator.get_next()
	images.set_shape([None, config.input_shape, config.input_shape, 3])
	bbox.set_shape([None, config.max_boxes, 5])
	grid_shapes = [config.input_shape // 32, config.input_shape // 16, config.input_shape // 8]
	draw_box(images, bbox)

	
	# Loading the pre-defined Graph
	if not os.path.exists(ckpt_path):
		os.mkdir(ckpt_path)
		convert(config.yolov3_cfg_path, config.yolov3_weights_path, ckpt_path, config.training, config.num_classes)
	
	else:
		if len(os.listdir(ckpt_path)) < 4:
			convert(config.yolov3_cfg_path, config.yolov3_weights_path, ckpt_path, config.training, config.num_classes)

	input_node, output, is_training, sess = load_graph(ckpt_path)

	graph = tf.get_default_graph()

	with graph.as_default():

		# moving_mean = tf.get_default_graph().get_tensor_by_name("convolutional_0/batch_norm/parameters/moving_mean:0")
		# moving_variance = tf.get_default_graph().get_tensor_by_name("convolutional_0/batch_norm/parameters/moving_variance:0")

		# Declaring the parameters for training the model
		with tf.name_scope('train_parameters'):
			epoch_loss = []
			global_step = tf.Variable(0, trainable=False, name='global_step')
			learning_rate = tf.train.exponential_decay(config.learning_rate, global_step,
				decay_steps, decay_rate)
			tf.summary.scalar('learning rate', learning_rate)

		# Declaring the parameters for GT
		with tf.name_scope('Targets'):
			bbox_true_13.set_shape([None, grid_shapes[0], grid_shapes[0], 3, 5 + config.num_classes])
			bbox_true_26.set_shape([None, grid_shapes[1], grid_shapes[1], 3, 5 + config.num_classes])
			bbox_true_52.set_shape([None, grid_shapes[2], grid_shapes[2], 3, 5 + config.num_classes])
		y_true = [bbox_true_13, bbox_true_26, bbox_true_52]


		x_reshape = tf.reshape(input_node, [-1, config.input_shape, config.input_shape, 3])

		# Compute Loss
		with tf.name_scope('Loss_and_Detect'):
			yolo_loss = compute_loss(output, y_true, anchors, config.num_classes, print_loss=False)
			tf.summary.scalar('YOLO_loss', yolo_loss)
			train_vars = tf.trainable_variables()

			# Variables to be optimized by train_op if the pre-trained darknet-53 is used as is
			variables = []
			if config.pre_train:
				if config.last_layers_only:
					variables.extend((train_vars[348], train_vars[349], train_vars[392], train_vars[393], train_vars[436], train_vars[437]))
				else:
					variables = train_vars[312:]
			else:
				variables = train_vars

			# for i, j in enumerate(variables):
			# 	print(i, j)
			# exit()

			# l2_loss = config.weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, dtype=tf.float32)) for v in variables]) 
			# loss = yolo_loss + l2_loss
			# tf.summary.scalar('L2_loss', l2_loss)

			loss = yolo_loss
			tf.summary.scalar('Total_loss', loss)

		# Define an optimizer for minimizing the computed loss
		with tf.name_scope('Optimizer'):
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

		with tf.control_dependencies(update_ops):
			train_op = optimizer.minimize(loss=loss, global_step=global_step, var_list=variables)

		# A saver object for saving the model
		saver = tf.train.Saver()

		# Getting all the summaries
		summary_op = tf.summary.merge_all()

		# Defining the summary writers for training and validation
		train_summary_writer = tf.summary.FileWriter(
			os.path.join(log_path, os.path.join(dataset, 'train')), sess.graph)
		# validation_summary_writer = tf.summary.FileWriter(
		# 	os.path.join(log_path, os.path.join(dataset, 'validation')), sess.graph)


		# Restoring the model
		if os.path.exists(os.path.join(ckpt_path, 'checkpoint')):
			if len(os.listdir(ckpt_path)) > 4:
				saver.restore(sess, checkmate.get_best_checkpoint(ckpt_path))
				print('Model Loaded!')
			# # Initializing all the defined tensorflow graph parameters 
			else:
				sess.run(tf.global_variables_initializer())
		else:
			sess.run(tf.global_variables_initializer())

		# Best checkpoint saver object
		best_ckpt_saver = BestCheckpointSaver(save_dir=ckpt_path, num_to_keep=5)

		for epoch in range(config.Epoch):

			mean_loss_train = []

			for k in tqdm(range(config.train_num//config.train_batch_size)):

				batch_image_train = sess.run(images)

				summary_train, loss_train, _ = sess.run([summary_op, loss,
					train_op], feed_dict={input_node: batch_image_train,
											is_training: True})

				train_summary_writer.add_summary(summary_train, epoch)
				train_summary_writer.flush()
				mean_loss_train.append(loss_train)

			mean_loss_train = np.mean(mean_loss_train)


			print('Epoch: {} completed\nTraining_loss: {}'.format(
				epoch+1, mean_loss_train))


			if epoch%5:
				best_ckpt_saver.handle(mean_loss_train, sess, global_step)

		print('Tuning Completed!!')
		train_summary_writer.close()
		# validation_summary_writer.close()
		sess.close()








def main(args):
	""" main function which calls all the other required functions for training
		Input:
			args : argument parser object containing the required command line arguments
	"""
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
	train(args.checkpoint_dir, args.logs_dir, 
		args.tfrecords_dir, args.dataset, args.class_path)



if __name__ == '__main__':
	main(parser.parse_args())