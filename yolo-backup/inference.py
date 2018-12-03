import warnings
warnings.filterwarnings("ignore")
from converter import convert
from utils.checkmate import BestCheckpointSaver
from utils import checkmate
import tensorflow as tf
import os
import cv2
import numpy as np
import argparse
from utils.utils import *
import matplotlib.pyplot as plt
from time import time
from PIL import Image, ImageFont, ImageDraw
import colorsys
import random


# Some command line arguments for running the model
parser = argparse.ArgumentParser(description="Run inference using darknet converted model")
parser.add_argument('checkpoint_path', help="Path to the converted checkpoints file")
parser.add_argument('img_path', help="Path for running inference on a single image or \
	multiple images")
parser.add_argument("dataset", help="Dataset for anchor and label selection")
parser.add_argument("output_path", help="Output Path to save the results")
parser.add_argument('--gpu-num', help="GPU to be used for running the inference", default=0)
parser.add_argument('--anchors_path', help="Path to the anchors for the model being used for\
	inference", default='./')
parser.add_argument('--labels_path', help="Path to the labels of dataset being used for \
	predicting labels", default='./model_data/')
parser.add_argument('--score_threshold', help='Threhold for considring TP', default=0.7)
parser.add_argument('--iou_threshold', help='Threshld IoU for applying NMS', default=0.2)
parser.add_argument('--max_boxes', help='Maximum number of boxes per image', default=20)



def read_image(img_path):
	""" A function which reads image(s) from the path provided
		Input:
			img_path: Path containing images
		Output:
			A batch containing all the images read using opencv
	"""
	print(img_path)
	assert img_path != None, 'Image path required for making inference'
	if os.path.exists(img_path):
		if os.path.isdir(img_path):
			img_dir = sorted(os.listdir(img_path))
			print('Reading {} images'.format(len(img_dir)))
			image = []
			for i in img_dir:
				img = cv2.imread(os.path.join(img_path, i))
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				image.append(img)
			print('Read {} images'.format(len(img_dir)))

		else:
			img = cv2.imread(img_path)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return image
	else:
		print("Path does not exists!!")


def get_classes(labels_path, dataset):
	""" Loads the classes 
		Input:
			labels_path: path in which dataset is kept
			dataset: dataset for getting the class names
		Output: list containing class names
	"""

	if dataset == 'coco':
		with open(labels_path + 'coco_classes.txt') as f:
			class_names = f.readlines()
	elif dataset == 'voc':
		with open(labels_path + 'voc_classes.txt') as f:
			class_names = f.readlines()
	else:
		with open(labels_path + 'raccoon_classes.txt') as f:
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


def load_graph(checkpoint_path):
	""" Loads the tensorflow saved graph for further processing.
		Input:
			checkpoint_path: string, path of the checkpoint of tensorflow graph
		Output:
			input_node: placeholder for feeding image into the graph
			output_nodes: output nodes of the graph
	"""

	# Restoring model
	# GPU memory allocation as per the model needs
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	meta_graph_path = [i for i in os.listdir(checkpoint_path) if i.endswith('.meta') ]
	tf.train.import_meta_graph(os.path.join(checkpoint_path, meta_graph_path[0]))
	saver = tf.train.Saver()
	if len(os.listdir(checkpoint_path))>4:
		print('Loading best checkpoint....')
		saver.restore(sess, checkmate.get_best_checkpoint(checkpoint_path))
		print('Best checkpoint: {} loaded'.format(checkmate.get_best_checkpoint(checkpoint_path)))
	else:
		latest_ckpt = tf.train.latest_checkpoint(checkpoint_path)
		saver.restore(sess, latest_ckpt)
	print('Model and classes loaded.')

	output_nodes = []

	# Retriving the input and output nodes for predictions
	input_node = tf.get_default_graph().get_tensor_by_name("Input:0")
	is_training = tf.get_default_graph().get_tensor_by_name("is_training:0")
	out_indices = tf.get_default_graph().get_tensor_by_name("Output_indices:0")
	out_indices_ = sess.run(out_indices, feed_dict={input_node: np.zeros((1, 416, 416, 3))})

	for i in out_indices_:
		# if retrain:
		# 	output_nodes.append(tf.get_default_graph().get_tensor_by_name(
		# 		"convolutional_"+str(i-1)+"/convolutional_"+str(i)+":0"))
		# else:
		# 	output_nodes.append(tf.get_default_graph().get_tensor_by_name(
		# 		"convolutional_"+str(i)+"/convolutional_"+str(i)+":0"))
		output_nodes.append(tf.get_default_graph().get_tensor_by_name(
			"convolutional_"+str(i)+"/convolutional_"+str(i)+":0"))

	return input_node, output_nodes, is_training, sess


def predict(output_nodes, anchors, num_classes, input_shape, image_shape, args):
	""" Predicts the output of an image
		Input:
			output_nodes: output_nodes of the graph
			anchors: list, anchor boxes used by the YOLO
			num_classes: int, number of classes for making predictions
			args: argparse object for retriving the required values
		Output:
			boxes: array, dimentions of the predicted boxes
			scores: array, scores corresponding to each box
			classes: array, classes corresponding to each box
	"""
	
	score_threshold = args.score_threshold
	iou_threshold = args.iou_threshold
	max_boxes = args.max_boxes
	num_output_layers = len(output_nodes)
	anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_output_layers==3 else [
		[3,4,5], [1,2,3]] # default setting
	boxes, box_scores = [], []


	for l in range(num_output_layers): # Making prediction for 3 scales
		_boxes, _box_scores = yolo_boxes_and_scores(output_nodes[l], 
													anchors[anchor_mask[l]], 
													num_classes, 
													input_shape, 
													image_shape)

		# list(3 arrays, 1 for each scale): [3, batch_size*grid_x*grid_y*3, 4]
		boxes.append(_boxes)
		# list(3 arrays, 1 for each scale): [3, batch_size*grid_x*grid_y*3, 80]
		box_scores.append(_box_scores)



	boxes = tf.concat(boxes, axis=0) # [3*batch_size*grid_x*grid_y, 4]
	box_scores = tf.concat(box_scores, axis=0) # [3*batch_size*grid_x*grid*y, 80]

	mask = box_scores >= score_threshold # True or False based on the box_scores
	# Maximum number of boxes to be selected by non max suppression
	max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)


	boxes_, scores_, classes_ = [], [], []

	for c in range(num_classes):

		"""
			Same thing applies to class_box_scores as well
			boxes: [3*batch_szie*grid_x*grid_y, 4], mask: [3*batch_size*grid_x*grid_y, 1]
			class_boxes: [..., 4], keep boxes which have (box_scores >= score_threshold)
		"""
		class_boxes = tf.boolean_mask(boxes, mask[:, c])
		class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

		# Apply the non max suppression after rejecting theboxes having box_scores lower than
		# a cretain threshold. This returns an integer tensor of indices having the shape [M<=20]
		nms_index = tf.image.non_max_suppression(class_boxes, # [num_boxes[True], 4]
												class_box_scores, #[num_boxes(True), 1]
												max_boxes_tensor, # default:20
												iou_threshold=iou_threshold,
												name='non_max_suppression')
		

		class_boxes = tf.gather(class_boxes, nms_index, 
			name='TopLeft_BottomRight') # Take the indexed elements (nms_index), shape:[M, 4]
		class_box_scores = tf.gather(class_box_scores, nms_index) # shape: [M, 1]
		
		classes = tf.ones_like(class_box_scores, dtype=tf.int32) * c
		boxes_.append(class_boxes)
		scores_.append(class_box_scores)
		classes_.append(classes)

	boxes = tf.concat(boxes_, axis=0)
	scores = tf.concat(scores_, axis=0)
	classes = tf.concat(classes_, axis=0)

	return boxes, scores, classes




def run_inference(checkpoint_path, img_path, dataset, args):
	""" A function making inference using the pre-trained darknet weights in the tensorflow 
		framework 
		Input:
			checkpoint_path: Path to the directory containing the pre-trained darknet weights 
				convertd to tensorflow compatible format using converter.py script.
			img_path: path to the image on which inference is to be run, path to the image 
				directory containing images in the case of multiple images.

		Output:
			YET TO BE DECIDED
	"""

	# Reading the images
	images_batch = read_image(img_path)

	out_path = args.output_path


	# Getting anchors and labels for the prediction
	class_names = get_classes(args.labels_path, dataset)
	anchors = read_anchors(os.path.join(args.anchors_path, 'yolo_anchors.txt'))


	input_node, output_nodes, is_training, sess = load_graph(checkpoint_path)
	num_output_layers = len(output_nodes)

	num_classes = len(class_names)
	num_anchors = anchors.shape[0]


	# Retriving the input shape of the model i.e. (608x608), (416x416), (320x320)
	input_shape = tf.shape(output_nodes[0])[1:3] * 32


	# Generate colors for drawing bounding boxes.
	hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
	random.seed(10101)  # Fixed seed for consistent colors across runs.
	random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
	random.seed(None)  # Reset seed to default.


	for x in range(len(images_batch)):
	
		image = images_batch[x]
		image_shape = (image.shape[0], image.shape[1])

		tick = time()
		# If the image size is not divisible by 32, it will be padded to make so
		new_image_size = (image.shape[0] - (image.shape[0] % 32), 
							image.shape[1] - (image.shape[1] % 32))
		image_data = np.array(resize_image(image, new_image_size))
		print('Image height: {}\tImage width: {}'.format(image.shape[0], image.shape[1]))
		img = image_data/255.
		img = np.expand_dims(img, 0) # Adding the batch dimention


		# Actually run the graph in a tensorflow session to get the outputs

		out_boxes, out_scores, out_classes = sess.run(predict(output_nodes, anchors, num_classes, 
			input_shape, image_shape, args), feed_dict={input_node: img, is_training: True})

		tock = time()

		print('Found {} boxes for {} in {}sec'.format(len(out_boxes), 'img', tock-tick))

		######################## Visualisation ######################
		font = ImageFont.truetype(font='./font/FiraMono-Medium.otf', 
			size=np.floor(3e-2 * image.shape[1] + 0.5).astype(np.int32))
		thickness = (image.shape[0] + image.shape[1]) // 500  # do day cua BB

		image = Image.fromarray((image).astype('uint8'), mode='RGB')
		daa = open(out_path+str(x)+'.txt', 'a')
		for i, c in reversed(list(enumerate(out_classes))):
			predicted_class = class_names[c]
			box = out_boxes[i]
			score = out_scores[i]

			label = '{} {:.2f}'.format(predicted_class, score)
			draw = ImageDraw.Draw(image)
			label_size = draw.textsize(label, font)
			# print(label_size)

			top, left, bottom, right = box  # y_min, x_min, y_max, x_max
			top = max(0, np.floor(top + 0.5).astype(np.int32))
			left = max(0, np.floor(left + 0.5).astype(np.int32))
			bottom = min(image.size[1], np.floor(bottom + 0.5).astype(np.int32))
			right = min(image.size[0], np.floor(right + 0.5).astype(np.int32))
			print(label, (left, top), (right, bottom))  # (x_min, y_min), (x_max, y_max)
			daa.write(str(label)+' '+str(left)+' '+str(top)+' '+str(right)+' '+str(bottom)+'\n')

			if top - label_size[1] >= 0:
				text_origin = np.array([left, top - label_size[1]])
			else:
				text_origin = np.array([left, top + 1])

			# My kingdom for a good redistributable image drawing library.
			for j in range(thickness):
				draw.rectangle([left + j, top + j, right - j, bottom - j], outline=colors[c])
			draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
			draw.text(text_origin, label, fill=(0, 0, 0), font=font)
			del draw

		# image.show()
		image.save(os.path.join(out_path, str(x)+'.png'))

		daa.close()

	sess.close()





		


def main(args):
	""" A function fetching the image data from the provided patha nd calling function 
		run_inference for doing the inference
		Input:
			args : argument parser object containing the required command line arguments
	"""
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
	run_inference(args.checkpoint_path, args.img_path, args.dataset, args)


if __name__ == '__main__':
	main(parser.parse_args())
