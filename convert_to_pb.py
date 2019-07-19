import tensorflow as tf
import config
from build_model import yolo, load_weights
from utils import checkmate
import numpy as np
from utils.utils import *
from tensorflow.python.tools import inspect_checkpoint as chkp
import os

def freeze_graph(sess, pb_output_path):
	terminal_node_names = ['input_image', 'input_shape', 'scale_1', 'scale_2', 'scale_3']

	terminal_node_names = ','.join(terminal_node_names)

	output_graph_def = tf.graph_util.convert_variables_to_constants(sess, 
		tf.get_default_graph().as_graph_def(), terminal_node_names.split(','))

	with tf.gfile.GFile(pb_output_path, 'wb') as f:
		f.write(output_graph_def.SerializeToString())

	print('{} ops written to {}'.format(len(output_graph_def.node), pb_output_path))


def get_classes(labels_path):
	""" Loads the classes 
		Input:
			labels_path: path in which classes.txt is kept
		Output: list containing class names
	"""


	with open(labels_path) as f:
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


def convert_model():

	# Getting anchors and labels for the prediction
	class_names = get_classes(config.classes_path)

	anchors = read_anchors(config.anchors_path)

	num_classes = config.num_classes
	num_anchors = config.num_anchors
	# Retriving the input shape of the model i.e. (608x608), (416x416), (320x320)
	input_shape = (config.input_shape, config.input_shape)


	# Defining placeholder for passing the image data onto the model
	image_tensor = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='input_image')
	image_shape = tf.placeholder(dtype=tf.int32, shape=[2], name='input_shape')

	output_nodes = yolo(input_images=image_tensor, is_training=False, config_path=config.yolov3_cfg_path, num_classes=config.num_classes)

	print(output_nodes)

	sess = tf.Session()

	scale_1, scale_2, scale3 = tf.identity(output_nodes[0], name='scale_1'), tf.identity(output_nodes[1], name='scale_2'), tf.identity(output_nodes[2], name='scale_3')

	ckpt_path = config.model_dir
	exponential_moving_average_obj = tf.train.ExponentialMovingAverage(config.weight_decay)
	saver = tf.train.Saver(exponential_moving_average_obj.variables_to_restore())
	ckpt = tf.train.get_checkpoint_state(ckpt_path)

	# chkp.print_tensors_in_checkpoint_file(checkmate.get_best_checkpoint(ckpt_path), tensor_name='', all_tensors=True)
	# exit()
	if config.pre_train is True:
		load_ops = load_weights(tf.global_variables(), config.yolov3_weights_path)
		sess.run(load_ops)
	elif ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print('Restoring model ', checkmate.get_best_checkpoint(ckpt_path))
		saver.restore(sess, checkmate.get_best_checkpoint(ckpt_path))
		print('Model Loaded!')
	else:
		print("No appropriate weights found for creating protobuf file")

	if not os.path.exists(config.model_export_path.split('/')[1]):
		os.mkdir(config.model_export_path.split('/')[1])

	freeze_graph(sess, config.model_export_path)

	sess.close()

def main():
	""" A function fetching the image data from the provided patha nd calling function 
		run_inference for doing the inference
		Input:
			args : argument parser object containing the required command line arguments
	"""
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_num)
	convert_model()

if __name__ == '__main__':
	main()
