""" Converts the darknet weight and config file for tensorflow """

# Importing stuff
import warnings
warnings.filterwarnings("ignore")
import argparse
import configparser
import tensorflow as tf
import config
import io
import os
from collections import defaultdict
import numpy as np
import sys
import cv2

# Some required command line arguments
parser = argparse.ArgumentParser(description="Darknet to Tensorflow Converter")
parser.add_argument('config_path', help="Path to the darknet cfg file")
parser.add_argument('weights_path', help="Path to the darknet trained weights file")
parser.add_argument('output_path', default='./converted/', 
	help="Path for saving converted weights")
parser.add_argument('--gpu-num', help="GPU to be used for running the inference", default=0)



def unique_config_sections(cfg_file):
	""" Converts the same names sections in darknet config file to unique sections
	Input:
		cfg_file: path of darknet .cfg file
	Output:
		cfg file having unique section names
	"""

	section_count = defaultdict(int)
	output_stream = io.StringIO()
	with open(cfg_file) as cfg:
		for line in cfg:
			if line.startswith('['):
				section_name = line.strip().strip('[]')
				new_section_name = section_name + '_' + str(section_count[section_name])
				section_count[section_name] += 1
				line = line.replace(section_name, new_section_name)
			output_stream.write(line)
	
	output_stream.seek(0)
	return output_stream



def convert(config_path, weights_path, output_path, retrain=False, num_classes=None):
	""" A function which converts the darknet weights to tensorflow compatible format
	Input:
		config_path: path of the darknet cfg file
		weights_path: path of the darknet weights corresponding to config_path
		output_path: path for saving the converted model
		retrain: weather or not the model is to be retrained
		num_classes: if retrain is True, the classes of the new dataset
	"""

	# Extracting the path from the argparse object
	config_path = os.path.expanduser(config_path)
	weights_path = os.path.expanduser(weights_path)
	output_path = os.path.expanduser(output_path)

	if not os.path.exists(output_path):
		os.mkdir(output_path)

	# Checking the provided files are right
	assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(config_path)
	assert weights_path.endswith('.weights'), '{} is not a .weights file'.format(weight_path)


	# Loading weights
	print("Parsing darknet weights....")
	weights_file = open(weights_path, 'rb')
	major_version, minor_version, revision = np.ndarray(shape=(3, ), dtype='int32', 
		buffer=weights_file.read(12))
	if (major_version*10+minor_version)>=2 and major_version<1000 and minor_version<1000:
		images_seen = np.ndarray(shape=(1, ), dtype='int32', buffer=weights_file.read(8))
	else:
		images_seen = np.ndarray(shape=(1, ), dtype='int32', buffer=weights_file.read(4))
	print("Weights header\nMajor version: {}\nMinor version: {}\nRevision: {}\nImages seen: {}"
		.format(major_version, minor_version, revision, images_seen))



	# Loading weight and config
	print("Parsing darknet config file....")
	unique_config_file = unique_config_sections(config_path)
	cfg_parser = configparser.ConfigParser()
	cfg_parser.read_file(unique_config_file)
	print("Config file read successfully!")

	model = config_path.split('/')[-1]
	model = model.split('.')[0]

	# fe_conv_count = 52

	if model=='yolov3':
		out_layers = [58, 66, 74]
	elif model=='yolov3-spp':
		out_layers = [59, 67, 75]
	elif model=='yolov3-tiny':
		out_layers = [9, 12, 12]

	ResizeMethod = tf.image.ResizeMethod()
	graph = tf.Graph()
	with graph.as_default():
		print("Trying to bulid tensorflow model...")

		width = int(cfg_parser['net_0']['width'])
		height = int(cfg_parser['net_0']['height'])

		weight_decay = float(cfg_parser['net_0']['decay']
			) if 'net_0' in cfg_parser.sections() else 5e-4

		momentum = float(cfg_parser['net_0']['momentum']
			) if 'net_0' in cfg_parser.sections() else 0.99

		initializer = tf.contrib.layers.xavier_initializer()
		regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

		input_layer = tf.placeholder(dtype='float32', shape=[None, None, None, 3], name='Input')
		is_training = tf.placeholder(tf.bool, name="is_training")
		prev_layer = input_layer
		all_layers = []

		count = 0
		out_index = []
		layer_count = -1
		for section in cfg_parser.sections():
			print("Parsing section {}".format(section))

			# Parsing Convolution layer
			if section.startswith('convolutional'):
				layer_count += 1
				# Retriving all the required information from the cfg file
				filters = int(cfg_parser[section]['filters']) # Number of conv kernels
				size = int(cfg_parser[section]['size']) # Size of the conv kernels
				stride = int(cfg_parser[section]['stride']) # Stride used for doing convolution
				pad = int(cfg_parser[section]['pad']) # If padding has been used or not
				activation = cfg_parser[section]['activation'] # Activation used, if any
				batch_normalization = 'batch_normalize' in cfg_parser[section] 

				# Setting padding for tensorflow according to the darknet cfg file
				padding = 'SAME' if pad == 1 and stride == 1 else 'VALID' 


				# Assigning the weights
				# Darknet serializes convolutional weights as :
				#	[bias/beta, [gamma, mean, variance], conv_weights]
				prev_layer_shape = prev_layer.get_shape().as_list()

				if (((layer_count==out_layers[0]) or (layer_count==out_layers[1]) or 
					(layer_count==out_layers[2])) and retrain):
						weights_shape_ = (size, size, prev_layer_shape[-1], 3*(num_classes+5))


				weights_shape = (size, size, prev_layer_shape[-1], filters)
				darknet_weights_shape = (filters, weights_shape[2], size, size)
				weights_size = np.prod(weights_shape)

				if (((layer_count==out_layers[0]) or (layer_count==out_layers[1]) or 
					(layer_count==out_layers[2])) and retrain):
						print('conv2d', 'bn' if batch_normalization else ' ', activation, 
						weights_shape_)
				else:
					print('conv2d', 'bn' if batch_normalization else ' ', activation, 
					weights_shape)


				conv_bias = np.ndarray(shape=(filters, ), dtype='float32', 
					buffer=weights_file.read(filters*4))
				count += filters

				if batch_normalization:
					bn_weights = np.ndarray(shape=(3, filters), dtype='float32', 
						buffer=weights_file.read(filters*4*3))
					count += 3 * filters

					bn_weight_list = [
						bn_weights[0], # Scale gamma
						conv_bias, # Shift beta - offset
						bn_weights[1], # Running mean
						bn_weights[2], # Running variance
						]


				conv_weights = np.ndarray(shape=darknet_weights_shape, dtype='float32', 
					buffer=weights_file.read(weights_size*4))
				count += weights_size


				# Darknet conv_weights are serialized like: (out_dim, in_dim, height, width)
				# In Tensorflow, we use weights like: (height, width, in_dim, out_dim)
				conv_weights = np.transpose(conv_weights, [2, 3, 1, 0]) # Changing the order
				print(conv_weights.shape, conv_bias.shape, sep='\n')
				# conv_weights = conv_weights if batch_normalization else [conv_weights, 
				# 	conv_bias]

				print(conv_weights.shape)

				with tf.name_scope(section):
					with tf.variable_scope('parameters', reuse=tf.AUTO_REUSE,
						regularizer=regularizer):
						if (((layer_count==out_layers[0]) or (layer_count==out_layers[1]) or 
							(layer_count==out_layers[2])) and retrain):
								weights_bias = tf.Variable(tf.random_normal([3*(num_classes+5)], 
									dtype=tf.float32), name=section+'_bias'+str(count))
								weights_conv = tf.Variable(initializer(weights_shape_, 
									dtype=tf.float32), name=section+'_weights'+str(count))
						else:
							weights_conv = tf.Variable(np.array(conv_weights))
							conv_bias = tf.Variable(conv_bias)
							# tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights_bias)
							tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights_conv)

					# Handling activations
					activation_fn = None
					if activation == 'leaky':
						pass # Advanced activation will be added on later
					elif activation != 'linear':
						raise ValueError('Unknown activation function `{}` in section {}'.format(
							activation, section))


					# Create convulutional layer
					if stride > 1:
						# Darknet uses left and top padding instead of 'same' mode
						paddings = tf.constant([[0, 0], [1, 0], [1, 0], [0, 0]])
						prev_layer = tf.pad(prev_layer, paddings, mode='CONSTANT')


					
					conv_layer = tf.nn.conv2d(input=prev_layer, filter=weights_conv, 
						strides=[1, stride, stride, 1] , padding=padding, name=section)

					if batch_normalization:
						with tf.name_scope('batch_norm'):


							with tf.variable_scope('parameters', reuse=tf.AUTO_REUSE):
								# if (((layer_count==out_layers[0]) or (layer_count==out_layers[1]) or 
								# 	(layer_count==out_layers[2])) and retrain):
								# 	moving_mean = tf.Variable(shape=bn_weight_list[2].shape, 
								# 		initializer=tf.zeros_initializer(), name='moving_mean')
								# 	moving_variance = tf.Variable(shape=bn_weight_list[3].shape, 
								# 		initializer=tf.ones_initializer(), name='moving_variance')
								# 	beta = tf.Variable(shape=bn_weight_list[1].shape, 
								# 		initializer='beta', name='beta')
								# 	gamma = tf.Variable(shape=bn_weight_list[0], 
								# 		initializer=tf.ones_initializer(), name='gamma')
								# else:
								# 	moving_mean = tf.Variable(bn_weight_list[2],
								# 		name='moving_mean')
								# 	moving_variance = tf.Variable(bn_weight_list[3],
								# 		name='moving_variance')
								# 	beta = tf.Variable(bn_weight_list[1], name='beta')
								# 	gamma = tf.Variable(bn_weight_list[0], name='gamma')

								moving_mean = tf.Variable(bn_weight_list[2],
									name='moving_mean')
								moving_variance = tf.Variable(bn_weight_list[3],
									name='moving_variance')
								beta = tf.Variable(bn_weight_list[1], name='beta')
								gamma = tf.Variable(bn_weight_list[0], name='gamma')

							
							def train():
								mean, variance = tf.nn.moments(x=conv_layer,
									axes=[0, 1, 2])

								# Calculating the exponential moving average for inference time
								mv_mean = tf.assign(moving_mean,
									moving_mean*momentum + mean*(1-momentum))
								mv_variance = tf.assign(moving_variance,
									moving_variance*momentum + mean*(1-momentum))


								with tf.control_dependencies([mv_mean, mv_variance]):
									tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mv_mean)
									tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mv_variance)
									return tf.identity(tf.nn.batch_normalization(x=conv_layer, 
										mean=mean, variance=variance, 
										offset=beta, scale=gamma, 
										variance_epsilon=1e-3), name='batch_norm_train')
									# return tf.identity(mean), tf.identity(variance)



							def valtest():

								# print_stmt = tf.Print(moving_mean, [moving_mean])

								return tf.identity(tf.nn.batch_normalization(x=conv_layer, 
									mean=moving_mean, variance=moving_variance, 
									offset=beta, scale=gamma, 
									variance_epsilon=1e-3), name='batch_norm_valtest')

								# return tf.identity(moving_mean), tf.identity(moving_variance)



							conv_layer = tf.case([(is_training, train)], valtest)

							# conv_layer = tf.keras.backend.switch(is_training, train, valtest)
							# mean, variance = tf.cond(pred=is_training,
							# true_fn=train, 
							# false_fn=valtest)

							# conv_layer = tf.nn.batch_normalization(x=conv_layer, 
							# 			mean=mean, variance=variance, 
							# 			offset=beta, scale=gamma, 
							# 			variance_epsilon=1e-3)



					prev_layer = conv_layer

					if activation == 'linear':
						all_layers.append(prev_layer)
					elif activation == 'leaky':
						with tf.name_scope('activation'):
							act_layer = tf.nn.leaky_relu(prev_layer, alpha=0.1)
							prev_layer = act_layer
							all_layers.append(act_layer)


			# Parsing Route layer
			elif section.startswith('route'):
				ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
				layers = [all_layers[i] for i in ids]
				if len(layers) > 1:
					print('Concatenating route layers:', layers)

					with tf.name_scope(section):
						concatenate_layers = tf.concat(layers, axis=-1)
					all_layers.append(concatenate_layers)
					prev_layer = concatenate_layers
				else:
					skip_layer = layers[0] # Only one layer to route
					all_layers.append(skip_layer)
					prev_layer = skip_layer


			# Parsing Max-Pooling layer
			elif section.startswith('maxpool'):
				size = int(cfg_parser[section]['size'])
				stride = int(cfg_parser[section]['stride'])

				with tf.name_scope(section):
					max_pooling = tf.nn.max_pool(prev_layer, ksize=[1, size, size, 1], 
						strides=[1, stride, stride, 1], padding='SAME')
				all_layers.append(max_pooling)
				prev_layer = all_layers[-1]



			# Parsing Shortcut/Res Layer
			elif section.startswith('shortcut'):
				index = int(cfg_parser[section]['from'])
				activation = cfg_parser[section]['activation']
				assert activation == 'linear', 'Only linear activation is supported'
				
				with tf.name_scope(section):
					shortcut = tf.math.add(all_layers[index], prev_layer)
				all_layers.append(shortcut)
				prev_layer = all_layers[-1]
				# if (retrain and pretrain) and layer_count==fe_conv_count-1:
				# 	stop_grads = tf.stop_gradient(prev_layer, name='feature_extractor')
				# 	prev_layer = stop_grads
				# 	all_layers.append(stop_grads)


			# Parsing Upsampling layer
			elif section.startswith('upsample'):
				stride = int(cfg_parser[section]['stride'])
				assert stride == 2, 'Only stride=2 is supported'
				# prev_layer = prev_layer.get_shape().as_list()

				with tf.name_scope(section):
					upsampled = tf.keras.layers.UpSampling2D(size=stride)(prev_layer)
				all_layers.append(upsampled)
				prev_layer = all_layers[-1]


			# Parsing YOLO layer
			elif section.startswith('yolo'):
				out_index.append(layer_count)
				all_layers.append(None)
				prev_layer = all_layers[-1]


			elif section.startswith('net'):
				pass # This is what we've been parsing in all of the code above


			else:
				raise ValueError('Unsupported section header type: {}'.format(section))


		# Creating and saving the parsed model
		print("The Model parsed is having {} layers describes as follows:\n".format(
			len(all_layers)-1))
		for i in all_layers:
			print(i)

		remaining_weights = len(weights_file.read())/4
		weights_file.close()
		if remaining_weights > 0:
			print("Warning: {} unsued weights".format(remaining_weights))
			if not retrain:
				exit()


		if len(out_index)==0:
			out_index.append(len(all_layers)-1)
		# output_layers = [all_layers[i] for i in out_index]
		# print(output_layers)


		output_layers_index = tf.Variable(np.array(out_index), name='Output_indices')

		print(len(tf.global_variables()), len(tf.trainable_variables()))
		init_op = tf.global_variables_initializer()
		saver = tf.train.Saver(tf.trainable_variables(), graph)

		with tf.Session(graph=graph) as sess:
			sess.run(init_op)
			sess.run(output_layers_index)
			print('Read {} of {} from Darknet weights.'.format(count, count+
			remaining_weights))
			print("Model saved in {} directory".format(output_path))
			saver.save(sess, output_path+'model')
			tf.summary.FileWriter('./logs', graph=graph)



def main(args):
	""" main function responsible for calling all the required functions """
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
	convert(args.config_path, args.weights_path, args.output_path)


if __name__ == '__main__':
	main(parser.parse_args())
