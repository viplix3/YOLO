import tensorflow as tf
import cv2
from PIL import Image, ImageDraw

def resize_image(image_data, size):
	""" Resizes the image without changing the aspect ratio with padding, so that
		the image size is divisible by 32, as per YOLO requirement.
		Input:
			image_data: array, original image data
			size: tuple, size the image is to e resized into
		Output:
			image: array, image data after resizing the image
	"""

	original_height, original_width, _ = image_data.shape
	req_height, req_width = size

	scale = min(req_width/original_width, req_height/original_height)
	new_width = int(req_width*scale)
	new_height = int(req_height*scale)

	image = cv2.resize(image_data, (new_width, new_height), 
		interpolation=cv2.INTER_CUBIC)
	background = Image.new('RGB', (req_width, req_height), (128, 128, 128))
	print(size)
	z = ((req_width-new_width)//2, (req_height-new_height)//2)

	foreground = Image.fromarray((image).astype('uint8'), mode='RGB')
	background.paste(foreground, z)

	return background


def yolo_head(output, anchors, num_classes, input_shape, calc_loss=False):
	""" Converts the output tensor of YOLO to bounding boxes parameters 
		Input:
			output: list, list of tenors containing the output of YOLO at different scales
			anchors: array, containing the anchors for the model
			num_classes: int, number of classes form which prediction are to be made
			input_shape: shape of the input fed to the network
			clas_loss: boolean, True if lass is to be calculated, used during training
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

		# In YOLO the height index is the innermost iteration
		# All the numbers in comments assume the grid is of shape 13x13
		grid_y = tf.range(0, grid_shape[0]) # array[0, 1, 2, ......, 11, 12]
		grid_x = tf.range(0, grid_shape[1])
		grid_y = tf.reshape(grid_y, [-1, 1, 1, 1]) # shape: [13, 1, 1, 1]
		grid_x = tf.reshape(grid_x, [1, -1, 1, 1]) # shape: [1, 13, 1, 1]
		grid_y = tf.tile(grid_y, [1, grid_shape[1], 1, 1]) # [13, 1, 1, 1] ---> [13, 13, 1, 1]
		grid_x = tf.tile(grid_x, [grid_shape[0], 1, 1, 1]) # [1, 13, 1, 1] ---> [13, 13, 1, 1]
		grid = tf.concat([grid_x, grid_y], axis=-1) # shape: [13, 13, 1, 2]
		grid = tf.cast(grid, dtype=output.dtype) # change dtype

	# Reshaping the output tensor into the form:
	#	[batch_size, grid_x, grid_y, num_anchors, box_parameters+classes]
	output = tf.reshape(output, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes+5])

	# Applying activation for getting the correct outputs
	with tf.name_scope('Generate_Output'):
		# We apply activations on the top of the output feature map to get the outputs
		box_xy = tf.nn.sigmoid(output[..., :2], name='x_y') # [None, 13, 13, 3, 2] 
		box_wh = tf.exp(output[..., 2:4], name='w_h') # [None, 13, 13, 3, 2]
		box_confidence = tf.sigmoid(output[..., 4:5]) # [None, 13, 13, 3, 1]
		box_class_probs = tf.sigmoid(output[..., 5:]) # [None, 13, 13, 3, num_classes]

		# Adjust rpedictions to each spatial grid point and anchor size
		# NOTE: YOLO iterates over height before the width
		box_xy = (box_xy + grid) / tf.cast(grid_shape[::-1], # (x, y + grid)/13 ---> b/w (0, 1)
			dtype=output.dtype) # [None, 13, 13, 3, 2]
		box_wh = box_wh * anchors_tensor / tf.cast(input_shape[::-1], dtype=output.dtype)

	if calc_loss == True:
		return grid, output, box_xy, box_wh
	return box_xy, box_wh, box_confidence, box_class_probs



def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
	""" Rescales the boxes according to the image_shape from the input_shape
		Input:
			box_xy, array, xy coordinates of box_mid
			box_wh, array, width and height of the boc
			input_shape: shape of the input of the model
			image_shape: input image shape
		Output:
			boxes: array, containing [x_min, y_min, x_max, y_max]
	"""

	# Because YOLO iterated over height before width, moreover tf.image.non_max_suppression
	# expects the Bounding Box paameters in this format only
	box_yx = box_xy[..., ::-1]
	box_hw = box_wh[..., ::-1]
	input_shape = tf.cast(input_shape, dtype=box_yx.dtype)
	image_shape = tf.cast(image_shape, dtype=box_hw.dtype)

	# Getting the scale and offset of the bounding boxes for the image_size
	new_shape = tf.round(image_shape * tf.reduce_min(input_shape/image_shape))
	offset = (input_shape-new_shape)/2./input_shape
	scale = input_shape/new_shape

	# Changing the predictions
	box_yx = (box_yx - offset) * scale # Moving the coordinates by offset and scaling 
	box_hw *= scale # Scaling the dimentions of the offset

	box_mins = box_yx - (box_hw / 2.)
	box_maxes = box_yx + (box_hw / 2.)

	# # Git Implementation
	boxes = tf.concat([
		box_mins[..., 0:1],  # y_min
		box_mins[..., 1:2], # x_min
		box_maxes[..., 0:1], # y_max
		box_maxes[..., 1:2] , # x_max
		], axis=-1)


	# Scale boxes back to the original shape
	boxes = tf.multiply(boxes, tf.concat([image_shape, image_shape], axis=-1))

	return boxes


def yolo_boxes_and_scores(output, anchors, num_classes, input_shape, image_shape):
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
	
	boxes_xy, boxes_wh, box_conf, box_clas_prob = yolo_head(output, anchors, num_classes,
		input_shape)
	boxes = yolo_correct_boxes(boxes_xy, boxes_wh, input_shape, image_shape)
	boxes = tf.reshape(boxes, [-1, 4])
	box_scores = tf.multiply(box_conf, box_clas_prob)
	box_scores = tf.reshape(box_scores, [-1, num_classes])
	return boxes, box_scores