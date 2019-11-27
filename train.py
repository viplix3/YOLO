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


def train(ckpt_path, log_path, class_path):
	""" Function to train the model.
		ckpt_path: string, path for saving/restoring the model
		log_path: string, path for saving the training/validation logs
		class_path: string, path for the classes of the dataset
		decay_steps: int, steps after which the learning rate is to be decayed
		decay_rate: float, rate to carrying out exponential decay
	"""


	# Getting the anchors
	anchors = read_anchors(config.anchors_path)

	classes = get_classes(class_path)

	if anchors.shape[0] // 3 == 2:
		yolo_tiny = True
	else:
		yolo_tiny = False

	# Building the training pipeline
	graph = tf.get_default_graph()

	with graph.as_default():

		# Getting the training data
		with tf.name_scope('data_parser/'):
			train_reader = Parser('train', config.anchors_path, config.output_dir, 
				config.num_classes, input_shape=config.input_shape, max_boxes=config.max_boxes)
			train_data = train_reader.build_dataset(config.train_batch_size//config.subdivisions)
			train_iterator = train_data.make_one_shot_iterator()

			val_reader = Parser('val', config.anchors_path, config.output_dir, 
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

			if yolo_tiny:
				images, bbox, bbox_true_13, bbox_true_26 = tf.cond(pred=tf.equal(mode, 1), true_fn = train, false_fn = valid, name='train_val__data')
				grid_shapes = [config.input_shape // 32, config.input_shape // 16]
			else:
				images, bbox, bbox_true_13, bbox_true_26, bbox_true_52 = tf.cond(pred=tf.equal(mode, 1), true_fn = train, false_fn = valid, name='train_val_data')
				grid_shapes = [config.input_shape // 32, config.input_shape // 16, config.input_shape // 8]

			images.set_shape([None, config.input_shape, config.input_shape, 3])
			bbox.set_shape([None, config.max_boxes, 5])

			# image_summary = draw_box(images, bbox)

		# Extracting the pre-defined yolo graph from the darknet cfg file
		if not os.path.exists(ckpt_path):
			os.mkdir(ckpt_path)
		output = yolo(images, is_training, config.yolov3_cfg_path, config.num_classes)


		# Declaring the parameters for GT
		with tf.name_scope('Targets'):
			if yolo_tiny:
				bbox_true_13.set_shape([None, grid_shapes[0], grid_shapes[0], config.num_anchors_per_scale, 5 + config.num_classes])
				bbox_true_26.set_shape([None, grid_shapes[1], grid_shapes[1], config.num_anchors_per_scale, 5 + config.num_classes])
				y_true = [bbox_true_13, bbox_true_26]
			else:
				bbox_true_13.set_shape([None, grid_shapes[0], grid_shapes[0], config.num_anchors_per_scale, 5 + config.num_classes])
				bbox_true_26.set_shape([None, grid_shapes[1], grid_shapes[1], config.num_anchors_per_scale, 5 + config.num_classes])				
				bbox_true_52.set_shape([None, grid_shapes[2], grid_shapes[2], config.num_anchors_per_scale, 5 + config.num_classes])
				y_true = [bbox_true_13, bbox_true_26, bbox_true_52]


		# Compute Loss
		with tf.name_scope('Loss_and_Detect'):
			loss_scale, yolo_loss, xy_loss, wh_loss, obj_loss, noobj_loss, conf_loss, class_loss = compute_loss(output, y_true, anchors, config.num_classes, config.input_shape, 
				ignore_threshold=config.ignore_thresh)
			loss = yolo_loss 
			exponential_moving_average_op = tf.train.ExponentialMovingAverage(config.weight_decay).apply(var_list=tf.trainable_variables()) # For regularisation
			scale1_loss_summary = tf.summary.scalar('scale_loss_1', loss_scale[0], family='Loss')
			scale2_loss_summary = tf.summary.scalar('scale_loss_2', loss_scale[1], family='Loss')
			yolo_loss_summary = tf.summary.scalar('yolo_loss', yolo_loss, family='Loss')
			# total_loss_summary = tf.summary.scalar('Total_loss', loss, family='Loss')
			xy_loss_summary = tf.summary.scalar('xy_loss', xy_loss, family='Loss')
			wh_loss_summary = tf.summary.scalar('wh_loss', wh_loss, family='Loss')
			obj_loss_summary = tf.summary.scalar('obj_loss', obj_loss, family='Loss')
			noobj_loss_summary = tf.summary.scalar('noobj_loss', noobj_loss, family='Loss')
			conf_loss_summary = tf.summary.scalar('confidence_loss', conf_loss, family='Loss')
			class_loss_summary = tf.summary.scalar('class_loss', class_loss, family='Loss')


		# Declaring the parameters for training the model
		with tf.name_scope('train_parameters'):
			global_step = tf.Variable(0, trainable=False, name='global_step')

			def learning_rate_scheduler(learning_rate, scheduler_name, global_step, decay_steps=100):
				if scheduler_name == 'exponential':
					lr =  tf.train.exponential_decay(learning_rate, global_step,
						decay_steps, decay_rate, staircase=True, name='exponential_learning_rate')
					return tf.maximum(lr, config.learning_rate_lower_bound)
				elif scheduler_name == 'polynomial':
					lr =  tf.train.polynomial_decay(learning_rate, global_step,
						decay_steps, config.learning_rate_lower_bound, power=0.8, cycle=True, name='polynomial_learning_rate')
					return tf.maximum(lr, config.learning_rate_lower_bound)
				elif scheduler_name == 'cosine':
					lr = tf.train.cosine_decay(learning_rate, global_step,
						decay_steps, alpha=0.5, name='cosine_learning_rate')
					return tf.maximum(lr, config.learning_rate_lower_bound)
				elif scheduler_name == 'linear':
					return tf.convert_to_tensor(learning_rate, name='linear_learning_rate')
				else:
					raise ValueError('Unsupported learning rate scheduler\n[supported types: exponential, polynomial, linear]')


			if config.use_warm_up:
				learning_rate = tf.cond(pred=tf.less(global_step, config.burn_in_epochs * (config.train_num // config.train_batch_size)),
					true_fn=lambda: learning_rate_scheduler(config.init_learning_rate, config.warm_up_lr_scheduler, global_step),
					false_fn=lambda: learning_rate_scheduler(config.learning_rate, config.lr_scheduler, global_step, decay_steps=500))
			else:
				learning_rate = learning_rate_scheduler(config.learning_rate, config.lr_scheduler, global_step, decay_steps=2000)

			tf.summary.scalar('learning rate', learning_rate)


		# Define optimizer for minimizing the computed loss
		with tf.name_scope('Optimizer'):
			# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=config.momentum)
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
			# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=config.momentum)
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				grads = optimizer.compute_gradients(loss=loss)
				gradients = [(tf.placeholder(dtype=tf.float32, shape=grad[1].get_shape()), grad[1]) for grad in grads]
				optimizing_op = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)
				# optimizing_op = optimizer.minimize(loss=loss, global_step=global_step)
			
			with tf.control_dependencies([optimizing_op]):
				with tf.control_dependencies([exponential_moving_average_op]):
					train_op_with_mve = tf.no_op()
			train_op = train_op_with_mve



#################################### Training loop ############################################################
		# A saver object for saving the model
		best_ckpt_saver_train = checkmate.BestCheckpointSaver(save_dir=ckpt_path+'train/', num_to_keep=5)
		best_ckpt_saver_valid = checkmate.BestCheckpointSaver(save_dir=ckpt_path+'valid/', num_to_keep=5)
		summary_op = tf.summary.merge_all()
		summary_op_valid = tf.summary.merge([yolo_loss_summary, xy_loss_summary, wh_loss_summary, 
			obj_loss_summary, noobj_loss_summary, conf_loss_summary, class_loss_summary, scale1_loss_summary, scale2_loss_summary])

		init_op = tf.global_variables_initializer()


		
		# Defining some train loop dependencies
		gpu_config = tf.ConfigProto(log_device_placement=False)
		gpu_config.gpu_options.allow_growth = True
		sess = tf.Session(config=gpu_config)
		tf.logging.set_verbosity(tf.logging.ERROR)
		train_summary_writer = tf.summary.FileWriter(os.path.join(log_path, 'train'), sess.graph)
		val_summary_writer = tf.summary.FileWriter(os.path.join(log_path, 'val'), sess.graph)
		
		# Restoring the model
		ckpt = tf.train.get_checkpoint_state(ckpt_path+'valid/')
		if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
			print('Restoring model ', checkmate.get_best_checkpoint(ckpt_path+'valid/'))
			tf.train.Saver().restore(sess, checkmate.get_best_checkpoint(ckpt_path+'valid/'))
			print('Model Loaded!')
		elif config.pre_train is True:
			sess.run(init_op)
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
				all_grads_and_vars, total_grads = [], []
				for minibatch in range(config.train_batch_size // config.subdivisions):
					num_steps, train_summary, loss_train, grads_and_vars = sess.run([global_step, summary_op, loss,
						grads], feed_dict={is_training: True, mode: 1})

					all_grads_and_vars.append(grads_and_vars)

					train_summary_writer.add_summary(train_summary, epoch)
					train_summary_writer.flush()
					mean_loss_train.append(loss_train)
					trainbar.set_description('Train loss: %s' %str(loss_train))

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

			if (config.use_warm_up):
				if (num_steps > config.burn_in_epochs * (config.train_num // config.train_batch_size)):
					best_ckpt_saver_train.handle(mean_loss_train, sess, global_step)
					best_ckpt_saver_valid.handle(mean_loss_valid, sess, global_step)
				else:
					continue
			else:
				best_ckpt_saver_train.handle(mean_loss_train, sess, global_step)
				best_ckpt_saver_valid.handle(mean_loss_valid, sess, global_step)

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
