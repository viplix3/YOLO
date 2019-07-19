# Importing stuff
import warnings
warnings.filterwarnings("ignore")
import configparser
import tensorflow as tf
import io
import os
from collections import defaultdict
import numpy as np
import config

def unique_config_sections(cfg_file):
	""" Converts the same names sections in darknet config file to unique sections
		Input:
			cfg_file: string, path of darknet .cfg file
		Output:
			outut_stream: cfg file having unique section names
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



def yolo(input_images, is_training, config_path, num_classes):
	""" A function which builds tensorflow model using the provided cfg file.
		Input:
			input_images: tensor, image tensor that will be provided as input to the model
			is_training: python bool, for different behavious of batch_norm during training and testing
			config_path: string, path of the darknet cfg file
			num_classes: int, number of classes in the dataset
		Output:
			returns the output nodes of the YOLO
	"""

	# Extracting the path from the argparse object
	config_path = os.path.expanduser(config_path)

	# Checking the provided files are right
	assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(config_path)

	if not os.path.exists(config.yolov3_cfg_path):
		print('cfg file not found.....\nPlease get the yolov3.cfg file from https://github.com/pjreddie/darknet/blob/master/cfg/')
		exit()

	# Loading and config file
	print("Parsing darknet config file....")
	unique_config_file = unique_config_sections(config_path)
	cfg_parser = configparser.ConfigParser()
	cfg_parser.read_file(unique_config_file)
	print("Config file read successfully!")

	model = config_path.split('/')[-1]
	model = model.split('.')[0]


	if model=='yolov3':
		out_layers = [58, 66, 74]
	elif model=='yolov3-spp':
		out_layers = [59, 67, 75]
	elif model=='yolov3-tiny':
		out_layers = [9, 12, 12]

	ResizeMethod = tf.image.ResizeMethod()

	print("Trying to bulid tensorflow model...")

	weight_decay = config.weight_decay

	momentum = config.norm_decay

	initializer = tf.glorot_uniform_initializer() # xavier initializer for initializing convolutinal filter weights
	regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay) # l2 regularizer for avoiding overfitting

	input_layer = input_images
	prev_layer = input_layer
	all_layers = []

	feature_extractor_conv_count = config.feature_extractor_conv_count
	count = -1 # Counts the total layers
	out_index = []
	layer_count = -1 # Counts the convolutional layers
	scope = 'darknet53/'
	switch_scope = False

	for section in cfg_parser.sections():
		print("Parsing section {}".format(section))

		# if count<74:
		# 	scope = 'darknet53/'
		# else:
		# 	scope = 'yolo/'
		if layer_count == feature_extractor_conv_count-1 and switch_scope:
			scope = 'yolo/'

		if layer_count == feature_extractor_conv_count-1:
			switch_scope = True

		with tf.name_scope(scope):

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

				if ((layer_count==out_layers[0]) or (layer_count==out_layers[1]) or 
					(layer_count==out_layers[2])):
						weights_shape_ = (size, size, prev_layer_shape[-1], 3*(num_classes+5))


				weights_shape = (size, size, prev_layer_shape[-1], filters)

				if ((layer_count==out_layers[0]) or (layer_count==out_layers[1]) or 
					(layer_count==out_layers[2])):
						print('conv2d', 'bn' if batch_normalization else ' ', activation, 
						weights_shape_)
				else:
					print('conv2d', 'bn' if batch_normalization else ' ', activation, 
					weights_shape)



				with tf.name_scope(section):

					if stride > 1:
						# Darknet uses left and top padding instead of 'same' mode
						paddings = tf.constant([[0, 0], [1, 0], [1, 0], [0, 0]])
						prev_layer = tf.pad(prev_layer, paddings, mode='CONSTANT')

					if ((layer_count==out_layers[0]) or (layer_count==out_layers[1]) or 
					(layer_count==out_layers[2])):
						conv_layer = tf.layers.conv2d(inputs=prev_layer, filters=config.num_anchors_per_scale*(num_classes+5), kernel_size=size, strides=[stride, stride], kernel_initializer=initializer,
								padding=padding, kernel_regularizer=regularizer, use_bias=1-batch_normalization, name=scope+section)
					else:
						conv_layer = tf.layers.conv2d(inputs=prev_layer, filters=filters, kernel_size=size, strides=[stride, stride], kernel_initializer=initializer,
							padding=padding, kernel_regularizer=regularizer, use_bias=1-batch_normalization, name=scope+section)

					if batch_normalization:
						with tf.name_scope('batch_norm'):
							bn_layer = tf.layers.batch_normalization(inputs=conv_layer, momentum=momentum, epsilon=1e-5, training=is_training, name=scope+section+'/batch_norm')
						conv_layer = bn_layer


					prev_layer = conv_layer

					if activation == 'linear':
						all_layers.append(prev_layer)
					elif activation == 'leaky':
						with tf.name_scope('activation'):
							act_layer = tf.nn.leaky_relu(prev_layer, alpha=0.1)
							prev_layer = act_layer
							all_layers.append(act_layer)
				count += 1



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
				count += 1



			# Parsing Max-Pooling layer
			elif section.startswith('maxpool'):
				size = int(cfg_parser[section]['size'])
				stride = int(cfg_parser[section]['stride'])

				with tf.name_scope(section):
					max_pooling = tf.nn.max_pool(prev_layer, ksize=[1, size, size, 1], 
						strides=[1, stride, stride, 1], padding='SAME')
				all_layers.append(max_pooling)
				prev_layer = all_layers[-1]
				count += 1




			# Parsing Shortcut/Res Layer
			elif section.startswith('shortcut'):
				index = int(cfg_parser[section]['from'])
				activation = cfg_parser[section]['activation']
				assert activation == 'linear', 'Only linear activation is supported'
				
				with tf.name_scope(section):
					shortcut = tf.math.add(all_layers[index], prev_layer)
				all_layers.append(shortcut)
				prev_layer = all_layers[-1]
				count += 1


			# Parsing Upsampling layer
			elif section.startswith('upsample'):
				stride = int(cfg_parser[section]['stride'])
				assert stride == 2, 'Only stride=2 is supported'

				with tf.name_scope(section):
					upsampled = tf.image.resize_nearest_neighbor(prev_layer, [2*tf.shape(prev_layer)[1], 2*tf.shape(prev_layer)[1]], name=section)
				all_layers.append(upsampled)
				prev_layer = all_layers[-1]
				count += 1


			# Parsing YOLO layer
			elif section.startswith('yolo'):
				out_index.append(count)
				all_layers.append(None)
				prev_layer = all_layers[-1]
				count += 1


			elif section.startswith('net'):
				pass # This is what we've been parsing in all of the code above


			else:
				raise ValueError('Unsupported section header type: {}'.format(section))


	# Summary of the parsed model
	print("The Model parsed is having {} layers describes as follows:\n".format(
		len(all_layers)-1))
	for i in all_layers:
		print(i)


	if len(out_index)==0:
		out_index.append(len(all_layers)-1)
	output_layers = [all_layers[i] for i in out_index]
	print(output_layers)

	return output_layers


def load_weights(var_list, weights_file):
	""" Loads the weights for the darknet weights file into the tensorflow model operations.
		Input:
			var_list = tensorflow variables for which the weights are to be loaded
			weights_file = darknet weights file
		Output:
			assign_ops: list, a list of tf.assign operations for assigning the weights to the required variable
	"""
	if not os.path.exists(weights_file):
		print('darknet53 weights not found.....\n')
		if not os.path.exists('./darknet53.conv.74'):
			os.system('wget https://pjreddie.com/media/files/darknet53.conv.74')
		os.system('mv darknet53.conv.74 ./darknet_data/darknet53.weights')
	with open(weights_file, "rb") as fp:
		_ = np.fromfile(fp, dtype=np.int32, count=5)

		weights = np.fromfile(fp, dtype=np.float32)

	ptr = 0
	i = 0
	assign_ops = []
	print('loading {} weights into {} variables...............'.format(weights.shape, len(var_list)))
	while i < len(var_list) - 1:
		var1 = var_list[i]
		var2 = var_list[i + 1]


		# do something only if we process conv layer
		if 'convolutional' in var1.name.split('/')[-2]:
			# check type of next layer
			if 'batch_norm' in var2.name.split('/')[-2]:
				# load batch norm params
				gamma, beta, mean, var = var_list[i + 1:i + 5]
				batch_norm_vars = [beta, gamma, mean, var]
				for var in batch_norm_vars:
					shape = var.shape.as_list()
					num_params = np.prod(shape)
					var_weights = weights[ptr:ptr + num_params].reshape(shape)
					ptr += num_params
					assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

				# we move the pointer by 4, because we loaded 4 variables
				i += 4
			elif 'convolutional' in var2.name.split('/')[-2]:
				# load biases
				bias = var2
				bias_shape = bias.shape.as_list()
				bias_params = np.prod(bias_shape)
				bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
				ptr += bias_params
				assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

				# we loaded 1 variable
				i += 1
				# we can load weights of conv layer
			shape = var1.shape.as_list()
			num_params = np.prod(shape)

			var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
			# remember to transpose to column-major
			var_weights = np.transpose(var_weights, (2, 3, 1, 0))
			ptr += num_params
			assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
		i += 1
	print('weights loaded seccuessfully!!')

	return assign_ops