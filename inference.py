import warnings
warnings.filterwarnings("ignore")
from build_model import yolo, load_weights
from utils import checkmate
import tensorflow as tf
import config
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
parser.add_argument('img_path', help="Path for running inference on a single image or \
	multiple images")
parser.add_argument("output_path", help="Output Path to save the results")
parser.add_argument("--darknet_model", help="Weather to use pre-trained darknet yolov3")



def read_image(img_path):
	""" A function which reads image(s) from the path provided
		Input:
			img_path: Path containing images
		Output:
			A batch containing all the images read using opencv
	"""
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



def predict(output_nodes, anchors, num_classes, input_shape, image_shape):
	""" Predicts the output of an image
		Input:
			output_nodes: output_nodes of the graph
			anchors: list, anchor boxes used by the YOLO
			num_classes: int, number of classes for making predictions
			input_shape: tuple, input image size to the model
			image_shape: tuple, original image shape
		Output:
			boxes: array, dimentions of the predicted boxes
			scores: array, scores corresponding to each box
			classes: array, classes corresponding to each box
	"""
	
	score_threshold = config.score_threshold
	iou_threshold = config.nms_threshold
	max_boxes = config.max_boxes
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

	# putting nms on the cpu for better FPS
	with tf.device('/device:CPU:0'):
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
			class_boxes = tf.batch_gather(class_boxes, nms_index, 
				name='TopLeft_BottomRight') # Take the indexed elements (nms_index), shape:[M, 4]
			class_box_scores = tf.batch_gather(class_box_scores, nms_index) # shape: [M, 1]
			
			classes = tf.ones_like(class_box_scores, dtype=tf.int32) * c
			boxes_.append(class_boxes)
			scores_.append(class_box_scores)
			classes_.append(classes)


		boxes = tf.concat(boxes_, axis=0)
		scores = tf.concat(scores_, axis=0)
		classes = tf.concat(classes_, axis=0)

		return boxes, scores, classes




def run_inference(img_path, output_dir,  args):
	""" A function making inference using the pre-trained darknet weights in the tensorflow 
		framework 
		Input:
			img_path: string, path to the image on which inference is to be run, path to the image directory containing images in the case of multiple images.
			output_dir: string, directory for saving the output
			args: argparse object
	"""

	# Reading the images
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	if not os.path.exists(os.path.join(output_dir, 'images')):
		os.mkdir(os.path.join(output_dir, 'images'))
	if not os.path.exists(os.path.join(output_dir, 'labels')):
		os.mkdir(os.path.join(output_dir, 'labels'))

	output_dir_images = os.path.join(output_dir, 'images')
	output_dir_labels = os.path.join(output_dir, 'labels')


	file_names = sorted(os.listdir(img_path))
	images_batch = read_image(img_path)


	# Getting anchors and labels for the prediction
	class_names = get_classes(config.classes_path)

	anchors = read_anchors(config.anchors_path)

	num_classes = config.num_classes
	num_anchors = config.num_anchors


	# Retriving the input shape of the model i.e. (608x608), (416x416), (320x320)
	input_shape = (config.input_shape, config.input_shape)


	# Generate colors for drawing bounding boxes.
	hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
	random.seed(10101)  # Fixed seed for consistent colors across runs.
	random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
	random.seed(None)  # Reset seed to default.

	# Defining placeholder for passing the image data onto the model
	image_tensor = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
	image_shape = tf.placeholder(dtype=tf.int32, shape=[2])

	# Building the model for running inference
	output_nodes = yolo(input_images=image_tensor, is_training=False, config_path=config.yolov3_cfg_path, num_classes=num_classes)

	# Creating a session for running the model
	gpu_config = tf.ConfigProto(log_device_placement=False)
	gpu_config.gpu_options.allow_growth = True
	sess = tf.Session(config=gpu_config)


	boxes, scores, classes = predict(output_nodes, anchors, num_classes, 
		input_shape, image_shape)

	total_time_pred = []
	total_time_yolo = []
	for x in range(len(images_batch)):
	
		image = images_batch[x]
		new_image_size = (config.input_shape, config.input_shape)
		image_data = np.array(resize_image(image, new_image_size))
		print('Image height: {}\tImage width: {}'.format(image.shape[0], image.shape[1]))


		img = image_data/255.
		img = np.expand_dims(img, 0) # Adding the batch dimension


		# Loading the model/weights for running the model
		if x < 1:
			if args.darknet_model is not None:
				print('Loading pre-trained weights.....')
				if not os.path.exists(config.yolov3_weights_path):
					print('yolov3 weights not found.....\n')
					if not os.path.exists('./yolov3.weights'):
						os.system('wget https://pjreddie.com/media/files/yolov3.weights')
					os.system('mv yolov3.weights ./darknet_data/yolov3.weights')
				load_op = load_weights(tf.global_variables(), weights_file=config.yolov3_weights_path)
				sess.run(load_op)

			else:
				ckpt_path = config.model_dir
				saver = tf.train.Saver()
				ckpt = tf.train.get_checkpoint_state(ckpt_path)
				if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
					print('Restoring model ', checkmate.get_best_checkpoint(ckpt_path))
					saver.restore(sess, checkmate.get_best_checkpoint(ckpt_path))
					print('Model Loaded!')


		# tick = time()
		# sess.run(output_nodes, feed_dict={image_tensor: img, image_shape: [image.shape[0], image.shape[1]]})
		# tock = time()
		# print("Prediction time: ", tock-tick)
		# total_time_yolo.append(tock-tick)

		tick = time()
		# Actually run the graph in a tensorflow session to get the outputs
		out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes], feed_dict={image_tensor: img, image_shape: [image.shape[0], image.shape[1]]})
		tock = time()
		total_time_pred.append(tock-tick)


		print('Found {} boxes for {} in {}sec'.format(len(out_boxes), 'img', tock-tick))

	 	######################## Visualization ######################
		font = ImageFont.truetype(font='./font/FiraMono-Medium.otf', 
			size=np.floor(1e-2 * image.shape[1] + 0.5).astype(np.int32))
		thickness = (image.shape[0] + image.shape[1]) // 1000  # do day cua BB

		image = Image.fromarray((image).astype('uint8'), mode='RGB')
		output_labels = open(os.path.join(output_dir_labels, file_names[x].split(',')[0]+'.txt'), 'w')
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
			output_labels.write(str(left)+','+str(top)+','+str(right)+','+str(bottom)+','+str(c)+','+str(score)+'\n')

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
		image.save(os.path.join(output_dir_images, file_names[x]), compress_level=1)

		output_labels.close()

	sess.close()

	total_time_pred = sum(total_time_pred[1:])
	# total_time_yolo = sum(total_time_yolo[1:])
	print('FPS of model with post processing over {} images is {}'.format(len(images_batch)-1, (len(images_batch)-1)/total_time_pred))
	# print('FPS of model over {} images is {}'.format(len(images_batch)-1, (len(images_batch)-1)/total_time_yolo))



def main(args):
	""" A function fetching the image data from the provided patha nd calling function 
		run_inference for doing the inference
		Input:
			args : argument parser object containing the required command line arguments
	"""
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_num)
	run_inference(args.img_path, args.output_path, args)


if __name__ == '__main__':
	main(parser.parse_args())
