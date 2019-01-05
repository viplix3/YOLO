import argparse
import tensorflow as tf
from tqdm import tqdm
import time
import os
from utils.yolo_loss import compute_loss
from utils.utils import draw_box
from dataParser import Parser
from utils import checkmate
from build_model import yolo, load_weights
import numpy as np
import config


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


def train(ckpt_path, log_path, class_path, decay_steps=2000, decay_rate=0.8):
	""" Function to train the model.
		ckpt_path: string, path for saving/restoring the model
		log_path: string, path for saving the training/validation logs
		class_path: string, path for the classes of the dataset
		decay_steps: int, steps after which the learning rate is to be decayed
		decay_rate: float, rate to carrying out exponential decay
	"""


	# Getting the anchors
	anchors = read_anchors(config.anchors_path)
	if not os.path.exists(config.data_dir):
		os.mkdir(config.data_dir)

	classes = get_classes(class_path)

	# Building the training pipeline
	graph = tf.get_default_graph()

	with graph.as_default():

		# Getting the training data
		with tf.name_scope('data_parser/'):
			train_reader = Parser('train', config.data_dir, config.anchors_path, config.output_dir, 
				config.num_classes, input_shape=config.input_shape, max_boxes=config.max_boxes)
			train_data = train_reader.build_dataset(config.train_batch_size//config.subdivisions)
			train_iterator = train_data.make_one_shot_iterator()

			val_reader = Parser('val', config.data_dir, config.anchors_path, config.output_dir, 
				config.num_classes, input_shape=config.input_shape, max_boxes=config.max_boxes)
			val_data = val_reader.build_dataset(config.val_batch_size//config.subdivisions)
			val_iterator = val_data.make_one_shot_iterator()


			is_training = tf.placeholder(dtype=tf.bool, shape=[], name='train_flag') # Used for different behaviour of batch normalization
			mode = tf.placeholder(dtype=tf.int16, shape=[], name='mode_flag')


			def train():
				# images, bbox, bbox_true_13, bbox_true_26, bbox_true_52 = train_iterator.get_next()
				return train_iterator.get_next()
			def valid():
				# images, bbox, bbox_true_13, bbox_true_26, bbox_true_52 = val_iterator.get_next()
				return val_iterator.get_next()

			images, bbox, bbox_true_13, bbox_true_26, bbox_true_52 = tf.cond(pred=tf.equal(mode, 1), true_fn = train, false_fn = valid, name='train_val_cond')

			images.set_shape([None, config.input_shape, config.input_shape, 3])
			bbox.set_shape([None, config.max_boxes, 5])

			grid_shapes = [config.input_shape // 32, config.input_shape // 16, config.input_shape // 8]
			draw_box(images, bbox)



		# Extracting the pre-defined yolo graph from the darknet cfg file
		if not os.path.exists(ckpt_path):
			os.mkdir(ckpt_path)
		output = yolo(images, is_training, config.yolov3_cfg_path, config.num_classes)


		# Declaring the parameters for GT
		with tf.name_scope('Targets'):
			bbox_true_13.set_shape([None, grid_shapes[0], grid_shapes[0], 3, 5 + config.num_classes])
			bbox_true_26.set_shape([None, grid_shapes[1], grid_shapes[1], 3, 5 + config.num_classes])
			bbox_true_52.set_shape([None, grid_shapes[2], grid_shapes[2], 3, 5 + config.num_classes])
		y_true = [bbox_true_13, bbox_true_26, bbox_true_52]


		# Compute Loss
		with tf.name_scope('Loss_and_Detect'):
			yolo_loss = compute_loss(output, y_true, anchors, config.num_classes, print_loss=False)
			l2_loss = tf.losses.get_regularization_loss()
			loss = yolo_loss+l2_loss
			yolo_loss_summary = tf.summary.scalar('yolo_loss', yolo_loss)
			l2_loss_summary = tf.summary.scalar('l2_loss', l2_loss)
			total_loss_summary = tf.summary.scalar('Total_loss', loss)


		# Declaring the parameters for training the model
		with tf.name_scope('train_parameters'):
			epoch_loss = []
			global_step = tf.Variable(0, trainable=False, name='global_step')
			learning_rate = tf.train.exponential_decay(config.learning_rate, global_step,
				decay_steps, decay_rate)
			tf.summary.scalar('learning rate', learning_rate)


		# Define optimizer for minimizing the computed loss
		with tf.name_scope('Optimizer'):
			#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=config.momentum)
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				if config.pre_train:
					train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo')
					grads = optimizer.compute_gradients(loss=loss, var_list=train_vars)
					gradients = [(tf.placeholder(dtype=tf.float32, shape=grad[1].get_shape()), grad[1]) for grad in grads]
					gradients = gradients * config.subdivisions
					train_step = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)
				else:
					grads = optimizer.compute_gradients(loss=loss)
					gradients = [(tf.placeholder(dtype=tf.float32, shape=grad[1].get_shape()), grad[1]) for grad in grads]
					gradients = gradients * config.subdivisions
					train_step = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)



#################################### Training loop ############################################################
		# A saver object for saving the model
		best_ckpt_saver = checkmate.BestCheckpointSaver(save_dir=ckpt_path, num_to_keep=5)
		summary_op = tf.summary.merge_all()
		summary_op_valid = tf.summary.merge([yolo_loss_summary, l2_loss_summary, total_loss_summary])
		init_op = tf.global_variables_initializer()


		
		# Defining some train loop dependencies
		gpu_config = tf.ConfigProto(log_device_placement=False)
		gpu_config.gpu_options.allow_growth = True
		sess = tf.Session(config=gpu_config)
		tf.logging.set_verbosity(tf.logging.ERROR)
		train_summary_writer = tf.summary.FileWriter(os.path.join(log_path, 'train'), sess.graph)
		val_summary_writer = tf.summary.FileWriter(os.path.join(log_path, 'val'), sess.graph)
		
		# Restoring the model
		ckpt = tf.train.get_checkpoint_state(ckpt_path)
		if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
			print('Restoring model ', checkmate.get_best_checkpoint(ckpt_path))
			tf.train.Saver().restore(sess, checkmate.get_best_checkpoint(ckpt_path))
			print('Model Loaded!')
		elif config.pre_train is True:
			load_ops = load_weights(tf.global_variables(scope='darknet53'), config.darknet53_weights_path)
			sess.run(load_ops)
		else:
			sess.run(init_op)

		print('Uninitialized variables: ', sess.run(tf.report_uninitialized_variables()))


		epochbar = tqdm(range(config.Epoch))
		for epoch in epochbar:
			epochbar.set_description('Epoch %s of %s' % (epoch, config.Epoch))
			mean_loss_train = []
			mean_loss_valid = []

			trainbar = tqdm(range(config.train_num//config.train_batch_size))
			for k in trainbar:
				total_grad = []
				for minibach in range(config.subdivisions):
					train_summary, loss_train, grads_and_vars = sess.run([summary_op, loss,
						grads], feed_dict={is_training: True, mode: 1})
					total_grad += grads_and_vars

				feed_dict = {is_training: True, mode: 1}
				for i in range(len(gradients)):
					feed_dict[gradients[i][0]] = total_grad[i][0]
				# print(np.shape(feed_dict))

				_ = sess.run(train_step, feed_dict=feed_dict)
				train_summary_writer.add_summary(train_summary, epoch)
				train_summary_writer.flush()
				mean_loss_train.append(loss_train)
				trainbar.set_description('Train loss: %s' %str(loss_train))


			print('Validating.....')
			valbar = tqdm(range(config.val_num//config.val_batch_size))
			for k in valbar:

				val_summary, loss_valid = sess.run([summary_op_valid, loss], feed_dict={is_training: False, mode: 0})

				val_summary_writer.add_summary(val_summary, epoch)
				val_summary_writer.flush()
				mean_loss_valid.append(loss_valid)
				valbar.set_description('Validation loss: %s' %str(loss_valid))


			mean_loss_train = np.mean(mean_loss_train)
			mean_loss_valid = np.mean(mean_loss_valid)

			print('\n')
			print('Train loss after %d epochs is: %f' %(epoch+1, mean_loss_train))
			print('Validation loss after %d epochs is: %f' %(epoch+1, mean_loss_valid))
			print('\n\n')

			if ((epoch+1)%3) == 0:
				best_ckpt_saver.handle(mean_loss_valid, sess, tf.constant(epoch))

		print('Tuning Completed!!')
		train_summary_writer.close()
		val_summary_writer.close()
		sess.close()




def main():
	""" main function which calls all the other required functions for training """
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_num)
	train(config.model_dir, config.logs_dir, config.classes_path)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



if __name__ == '__main__':
	main()
