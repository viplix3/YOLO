""" Converts the darknet weight and config file for tensorflow """

# Importing stuff
import warnings
warnings.filterwarnings("ignore")
import argparse
import configparser
import tensorflow as tf
import io
import os
from collections import defaultdict
import numpy as np
import config
import sys
import cv2

# Some required command line arguments
parser = argparse.ArgumentParser(description="Darknet to Tensorflow Converter")
parser.add_argument('config_path', help="Path to the darknet cfg file")
parser.add_argument('weights_path', help="Path to the darknet trained weights file")
parser.add_argument('output_path', default='./converted/', 
	help="Path for saving converted weights")



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



def convert(args):
	""" A function which converts the darknet weights to tensorflow compatible format
	Input:
		args : argparse object containing the command line arguments provided by the user.
	"""

	# Extracting the path from the argparse object
	config_path = os.path.expanduser(args.config_path)
	weights_path = os.path.expanduser(args.weights_path)
	output_path = os.path.expanduser(args.output_path)

	if not os.path.exists(output_path):
		os.mkdir(output_path)

	# Checking the provided files are right
	assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(config_path)
	assert weights_path.endswith('.weights'), '{} is not a .weights file'.format(weight_path)

	if not os.path.exists(config_path):
		print('cfg file not found.....\nPlease get the yolov3.cfg file from https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg')
		exit()

	if not os.path.exists(config.yolov3_weights_path):
		print('darknet53 weights not found.....\n')
		if not os.path.exists('./yolov3.weights'):
			os.system('wget https://pjreddie.com/media/files/yolov3.weights')
		os.system('mv yolov3.weights ./darknet_data/yolov3.weights')


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

	graph = tf.Graph()
	with graph.as_default():
		print("Trying to bulid tensorflow model...")
		input_layer = tf.placeholder(dtype='float32', shape=[None, None, None, 3], name='Input')
		prev_layer = input_layer
		all_layers = []

		weight_decay = float(cfg_parser['net_0']['decay']
			) if 'net_0' in cfg_parser.sections() else 5e-4

		count = 0
		out_index = []
		for section in cfg_parser.sections():
			print("Parsing section {}".format(section))

			# Parsing Convolution layer
			if section.startswith('convolutional'):
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

				weights_shape = (size, size, prev_layer_shape[-1], filters)
				darknet_weights_shape = (filters, weights_shape[2], size, size)
				weights_size = np.prod(weights_shape)

				print('conv2d', 'bn' if batch_normalization else ' ', activation, 
					weights_shape)

				conv_bias = np.ndarray(shape=(filters, ), dtype='float32', 
					buffer=weights_file.read(filters*4))
				count += filters

				if batch_normalization:
					bn_weights = np.ndarray(shape=(3, filters), dtype='float32', 
						buffer=weights_file.read(filters*4*3))
					count += 3 * filters

					bn_weight_list = tf.Variable([
						bn_weights[0], # Scale gamma
						tf.Variable(conv_bias), # Shift beta - offset
						bn_weights[1], # Running mean
						bn_weights[2], # Running variance
						])

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

				weights_conv = tf.Variable(np.array(conv_weights))
				conv_bias = tf.Variable(conv_bias)

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


				# regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
				conv_layer = tf.nn.conv2d(input=prev_layer, filter=weights_conv, 
					strides=[1, stride, stride, 1] , padding=padding)

				if batch_normalization:
					conv_layer = tf.nn.batch_normalization(x=conv_layer, 
						mean=bn_weight_list[2], variance=bn_weight_list[3], 
						offset=bn_weight_list[1], scale=bn_weight_list[0], 
						variance_epsilon=1e-3)

					# conv_layer = tf.layers.batch_normalization(inputs=batch_normalization)

				prev_layer = conv_layer

				if activation == 'linear':
					all_layers.append(prev_layer)
				elif activation == 'leaky':
					act_layer = tf.nn.leaky_relu(prev_layer, alpha=0.1)
					prev_layer = act_layer
					all_layers.append(act_layer)


			# Parsing Route layer
			elif section.startswith('route'):
				ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
				layers = [all_layers[i] for i in ids]
				if len(layers) > 1:
					print('Concatenating route layers:', layers)
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
				max_pooling = tf.nn.max_pool(prev_layer, ksize=[1, size, size, 1], 
					strides=[1, stride, stride, 1], padding='SAME')
				all_layers.append(max_pooling)
				prev_layer = all_layers[-1]



			# Parsing Shortcut/Res Layer
			elif section.startswith('shortcut'):
				index = int(cfg_parser[section]['from'])
				activation = cfg_parser[section]['activation']
				assert activation == 'linear', 'Only linear activation is supported'
				shortcut = tf.math.add(all_layers[index], prev_layer)
				all_layers.append(shortcut)
				prev_layer = all_layers[-1]


			# Parsing Upsampling layer
			elif section.startswith('upsample'):
				stride = int(cfg_parser[section]['stride'])
				assert stride == 2, 'Only stride=2 is supported'
				
				upsampled = tf.image.resize_nearest_neighbor(prev_layer, [2*tf.shape(prev_layer)[1], 2*tf.shape(prev_layer)[1]], name=section)
				all_layers.append(upsampled)
				prev_layer = all_layers[-1]


			# Parsing YOLO layer
			elif section.startswith('yolo'):
				out_index.append(len(all_layers)-1)
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
			exit()


		if len(out_index)==0:
			out_index.append(len(all_layers)-1)
		output_layers = [all_layers[i] for i in out_index]
		print(output_layers)


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
	os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_num)
	convert(args)


if __name__ == '__main__':
	main(parser.parse_args())
