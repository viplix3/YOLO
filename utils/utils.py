import tensorflow as tf
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def resize_image(image_data, size):
	""" Resizes the image without changing the aspect ratio with padding, so that
		the image size is divisible by 32, as per YOLO requirement.
		Input:
			image_data: array, original image data
			size: tuple, size the image is to e resized into
		Output:
			image: array, image data after resizing the image
	"""

	image_height, image_width, _ = image_data.shape
	input_height, input_width = size

	# Getting the scale that is to be used for resizing the image
	scale = min(input_width / image_width, input_height / image_height)
	new_width = int(image_width * scale) # new image width
	new_height = int(image_height * scale) # new image height

	# getting the number of pixels to be padded
	dx = (input_width - new_width)
	dy = (input_height - new_height)

	# resizing the image
	image = cv2.resize(image_data, (new_width, new_height), 
		interpolation=cv2.INTER_CUBIC)


	top, bottom = dy//2, dy-(dy//2)
	left, right = dx//2, dx-(dx//2)

	color = [128, 128, 128] # color pallete to be used for padding
	new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color) # padding
	
	return new_image


def get_head(output, anchors, num_classes, input_shape, calc_loss=False):
	""" Converts the output tensor of YOLO to bounding boxes parameters 
		Input:
			output: list, list of tenors containing the output of YOLO at different scales
			anchors: array, containing the anchors for the model
			num_classes: int, number of classes form which prediction are to be made
			input_shape: shape of the input fed to the network
			clas_loss: boolean, True if loss is to be calculated, used during training
		Output:
			box_xy: array, center coordinated of the boxes
			box_wh: array, width and height of the boxes
			box_confidence: array, confidence associated with every box
			box_clas_prob: array, per class probablities for every box 
	"""

	num_anchors = anchors.shape[0]
	anchors_tensor = tf.reshape(tf.constant(anchors, dtype=output.dtype), 
		[1, 1, 1, num_anchors, 2])

	with tf.name_scope('Create_GRID'):
		grid_shape = tf.shape(output)[1:3] # Get the height and width of the grid
		# grid: [input_shape/32, input_shape/32] for scale 1
		# grid: [input_shape/16, input_shape/16] for scale 2
		# grid: [input_shape/8, input_shape/8] for scale 3

		grid_x = tf.range(grid_shape[1], dtype=tf.int32) # [0, 1, 2, ..., 12]
		grid_y = tf.range(grid_shape[0], dtype=tf.int32) # [0, 1, 2, ..., 12]
		meshed_x, meshed_y = tf.meshgrid(grid_x, grid_y)
		x_offset = tf.reshape(meshed_x, (-1, 1))
		y_offset = tf.reshape(meshed_y, (-1, 1))
		xy_offset = tf.concat([x_offset, y_offset], axis=-1)
		xy_offset = tf.reshape(xy_offset, [grid_shape[0], grid_shape[1], 1, 2])
		xy_offset = tf.cast(xy_offset, dtype=output.dtype)

	# Reshaping the output tensor into the form:
	#	[batch_size, grid_x, grid_y, num_anchors, box_parameters+classes]
	output = tf.reshape(output, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes+5])

	# Applying activation for getting the correct outputs
	with tf.name_scope('Generate_Output'):
		# We apply activations on the top of the output feature map to get the outputs
		box_xy = tf.nn.sigmoid(output[..., :2], name='x_y') # [None, 13, 13, 3, 2] 
		box_wh = tf.exp(output[..., 2:4], name='w_h') # [None, 13, 13, 3, 2]
		box_confidence = tf.nn.sigmoid(output[..., 4:5]) # [None, 13, 13, 3, 1]
		box_class_probs = tf.nn.sigmoid(output[..., 5:]) # [None, 13, 13, 3, num_classes]

		# Adjust pedictions to each spatial grid point and anchor size
		# NOTE: YOLO iterates over height before the width
		box_xy = (box_xy + xy_offset) * tf.cast(input_shape[::-1] // tf.cast(grid_shape[::-1], dtype=output.dtype), # absolute values [mid_x, mid_y] of predicted box
			dtype=output.dtype) # [None, 13, 13, 3, 2]
		box_wh = box_wh * anchors_tensor # absolute values [width, height] of predicted box

	if calc_loss == True:
		return xy_offset, output, box_xy, box_wh
	return box_xy, box_wh, box_confidence, box_class_probs



def correct_boxes(box_xy, box_wh, input_shape, image_shape):
	""" Rescales the boxes according to the image_shape from the input_shape
		Input:
			box_xy, array, xy coordinates of box_mid
			box_wh, array, width and height of the box
			input_shape: shape of the input of the model
			image_shape: input image shape
		Output:
			boxes: array, containing [y_min, x_min, y_max, x_max]
	"""

	input_shape = tf.cast(input_shape, dtype=box_xy.dtype)
	image_shape = tf.cast(image_shape, dtype=box_wh.dtype)

	scale = tf.reduce_min(input_shape/image_shape)
	new_shape = tf.round(image_shape * scale)
	offset = (input_shape-new_shape)

	box_mins = box_xy - (box_wh / 2.)
	box_maxes = box_xy + (box_wh / 2.)

	# [y_min, x_min, y_max, x_max] as we will be using tf.image.non_max_suppression
	boxes = tf.concat([
		(box_mins[..., 1:2] - offset[0]/2.) * (image_shape[0] / (input_shape[0] - offset[0])),  # y_min
		(box_mins[..., 0:1] - offset[1]/2.) * (image_shape[1] / (input_shape[1] - offset[1])), # x_min
		(box_maxes[..., 1:2] - offset[0]/2.) * (image_shape[0] / (input_shape[0] - offset[0])), # y_max
		(box_maxes[..., 0:1] - offset[1]/2.) * (image_shape[1] / (input_shape[1]  - offset[1])), # x_max
		], axis=-1)

	return boxes


def get_boxes_and_scores(output, anchors, num_classes, input_shape, image_shape):
	""" Computes the output of YOLO and returns boxes and their corresponding scores
		Input:
			output: tensorflow tensor, output nodes of the YOLO mode
			anchors: array, anchors used by the YOLO for given output node
			num_classes: int, number of classes for making predictions
			input_shape: tuple, shape of the input of the YOLO model
			image_shape: tuple, shape of the image fed to the YOLO for prediction
		Output:
			boxes: array, filtered boxes predicted by the YOLO
			box_scores: array, scores corresponding box in boxes
	"""
	
	boxes_xy, boxes_wh, box_conf, box_clas_prob = get_head(output, anchors, num_classes,
		input_shape)
	boxes = correct_boxes(boxes_xy, boxes_wh, input_shape, image_shape)
	boxes = tf.reshape(boxes, [-1, 4])

	best_class_scores = tf.reduce_max(box_clas_prob, axis=-1, keepdims=True)
	class_mask = tf.cast(box_clas_prob >= best_class_scores, dtype=tf.float32)

	class_scores = tf.multiply(box_clas_prob, class_mask)
	box_scores = tf.multiply(box_conf, class_scores)

	# print_op = tf.Print(class_mask, [tf.shape(class_mask)], message="alalalalala: ")

	# with tf.control_dependencies([print_op]):
	box_scores = tf.reshape(box_scores, [-1, num_classes])
	return boxes, box_scores


def draw_box(image, bbox):
	""" Draws boxes over the images provided for tensorboard.
		Input:
			image: tfrecord file holding the image information
			bbox: bounding box parameters
	"""

	with tf.name_scope('summary_image'):
		xmin, ymin, xmax, ymax, label = tf.split(value = bbox, num_or_size_splits = 5, axis=2)
		height = tf.cast(tf.shape(image)[1], tf.float32)
		weight = tf.cast(tf.shape(image)[2], tf.float32)
		new_bbox = tf.concat([tf.cast(ymin, tf.float32) / height, tf.cast(xmin, tf.float32) / weight, tf.cast(ymax, tf.float32) / height, tf.cast(xmax, tf.float32) / weight], 2)
		new_image = tf.image.draw_bounding_boxes(image, new_bbox)
		return tf.summary.image('image', tensor=new_image)