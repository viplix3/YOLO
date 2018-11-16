# Importing some necessary libraries to run the program
import tensorflow as tf
import numpy as np
import os
import sys
import threading
import random
from datetime import datetime
from operator import itemgetter

# Plotting libraries, for testing
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches



# Defining some flags
tf.app.flags.DEFINE_string('train_annotations', './train.txt', 
	'Train annotations for all the images')
tf.app.flags.DEFINE_string('val_annotations', './val.txt', 
	'Validation annotations for all images')
tf.app.flags.DEFINE_string('output_dir', './tfrecords/', 
	'Output directory for tfrecords')
tf.app.flags.DEFINE_string('classes_path', './model_data/raccoon_classes.txt', 
	'Path to the file containing the classes of dataset')
tf.app.flags.DEFINE_string('anchors_path', './yolo_anchors.txt', 
	'Path to the file containing anchor boxes')
tf.app.flags.DEFINE_string('dataset_dir', './dataset/', 
	'Parent directory for dataset')


tf.app.flags.DEFINE_integer('train_threads', 5, 
	'Number of threads to be used for processing training images')
tf.app.flags.DEFINE_integer('val_threads', 2, 
	'Number of threads to be used for processing validation images')
tf.app.flags.DEFINE_integer('train_shards', 10, 
	'Number of shards for training data')
tf.app.flags.DEFINE_integer('val_shards', 2, 
	'Number of shards for validation data')
tf.app.flags.DEFINE_integer('num_classes', 1, 
	'Number of object classes in the dataset')
tf.app.flags.DEFINE_integer('input_shape', 416,
	'YOLO input shape')
# tf.app.flags.DEFINE_integer('num_anchors_per_scale', 3, 
# 	'Number of anchors to be used for each output layer/scale')

FLAGS = tf.app.flags.FLAGS


# If there is no output folder for holding tfrecords, create one
if not os.path.exists(FLAGS.output_dir):
	os.mkdir(FLAGS.output_dir)



def _int64_feature(value):
	""" Converts the given input into an int64 feature that can be used in tfrecords
		Input:
			value: value to be converte into int64 feature
		Output:
			tf.train.Int64List object encoding the int64 value that can be used in tfrecords
	"""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def _bytes_feature(value):
	""" Converts the given input into a bytes feature that can be used in tfrecords
		Input:
			value: value to be converted into bytes feature
		Output:
			tf.train.BytesList object that can be used in tfrecords
	"""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



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



def read_annotations(file_path):
	""" Reads the image_path and annotations from train.txt
		Input:
			file_path: path to tran.txt contatining annotations
		Output:
			file_name: array, containing relative path of dataset files
			BB: array, containing Bouding Boxes coordinates for each file_name row
			class_id: class_id for each file_name row
	"""
	classes = get_classes(FLAGS.classes_path)
	file_name = []
	BB = []
	class_id = []
	with open(file_path) as file:
		for lines in file.read().splitlines():
			line = lines.split()
			name = os.path.join(FLAGS.dataset_dir, line[0])
			file_name.append(os.path.join(FLAGS.dataset_dir, line[0]))
			line = line[1::]
			_BB = []
			_class_id = []

			for i in range(len(line)):
				_BB.append(line[i].split(',')[:-1])
				_class_id.append(int(line[i].split(',')[-1]))


			BB.append(np.array(_BB, dtype='float32'))
			class_id.append(np.array(_class_id, dtype='int32'))

	return np.array(file_name), np.array(BB), np.array(class_id)



def process_tfrecord_batch(mode, thread_index, ranges, file_names, bb, classes, anchors):
	""" Processes images and saves tfrecords 
		Input:
			mode: string, specify if the tfrecords are to be made for training, validation 
				or testing
			thread_index: specifies the thread which is executing the function
			ranges: list, specifies the range of images the thread calling this function 
				will process
			file_path: array, containing the relative filepaths of images
			bb: array, containing bounding boxes of all the objects in an image
			classes: array, containing class_id associated to every bounding box
			anchors: array, anchors for the given dataset
	"""

	if mode == 'train':
		num_threads = FLAGS.train_threads
		num_shards = FLAGS.train_shards

	if mode == 'val' or mode == 'test':
		num_threads = FLAGS.val_threads
		num_shards = FLAGS.val_shards

	num_anchors = np.shape(anchors)[0]

	num_shards_per_batch = int(num_shards/num_threads)
	shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1], 
		num_shards_per_batch+1).astype(int)
	num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

	counter = 0
	for s in range(num_shards_per_batch):
		shard = thread_index * num_shards_per_batch + s
		output_filename = '%s-%.5d-of-%.5d.tfrecord' % (mode, shard, num_shards)
		output_file = os.path.join(FLAGS.output_dir, output_filename)
		writer = tf.python_io.TFRecordWriter(output_file)

		shard_count = 0
		files_in_shard = np.arange(shard_ranges[s], shard_ranges[s+1], dtype=int)
		
		for i in files_in_shard:

			_filename = file_names[i]
			_classes = classes[i]
			_bb = bb[i]

			image_data, width, height = _process_image(_filename)
			_bb = scaled_bounding_boxes(_bb, width, height)

			# IMAGE PLOTTING
			# fix, ax = plt.subplots()
			# ax.imshow(image_data)
			# for i in range(_bb.shape[0]):
			# 	x = _bb[i][0]
			# 	y = _bb[i][1]
			# 	x_max = _bb[i][2]
			# 	y_max = _bb[i][3]
			# 	print(x, y, x_max, y_max)
			# 	rect = patches.Rectangle((x, y), (x_max-x), (y_max-y), 
			# 	linewidth=2, edgecolor='r', 
			# 		facecolor='none')
			# 	ax.add_patch(rect)
			# 	ax.text(x, y, _classes[i], horizontalalignment='left', 
			# 		verticalalignment='bottom', color='b')
			# 	plt.axis('off')
			# plt.show()

			
			label = create_labels(_bb, _classes, anchors)

			example = convert_to_example(image_data, label)
			
			writer.write(example.SerializeToString())
			shard_count += 1
			counter += 1

		
		writer.close()
		print('%s [thread %d]: Wrote %d images to %s' % (datetime.now(), thread_index, 
			shard_count, output_file))
		shard_count = 0
	print('%s [thread %d]: Wrote %d images to %d shards.' % (datetime.now(), thread_index, 
		counter, num_files_in_thread))



def _process_image(filename):
	""" Read image files from disk 
		Input:
			file_name: str, relative path of the image
		Output:
			img_data: array, containing the image data
			width: width of the read image
			height: height of the read image
	"""
	# print(filename)
	img_data = cv2.imread(filename)
	width, height, _ = img_data.shape
	img_data = cv2.resize(img_data, (FLAGS.input_shape, FLAGS.input_shape))
	img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

	return img_data, width, height



def scaled_bounding_boxes(bb, width, height):
	""" Scales the bounding boxes for the rescaled image 
		Input:
			bb: array, original bounding boxes
			width: original width of the image
			height: original height of the image
		Output:
			rescaled bounding boxes according to the image dimentions
	"""
	for i in range(bb.shape[0]):

		_x_min, _y_min, _x_max, _y_max = bb[i][0], bb[i][1], bb[i][2], bb[i][3]
		x_scale = (FLAGS.input_shape / height)
		y_scale = (FLAGS.input_shape / width)

		bb[i][0] = max(int(np.round(_x_min * x_scale)), 0)
		bb[i][1] = max(int(np.round(_y_min * y_scale)), 0)
		bb[i][2] = min(int(np.round(_x_max * x_scale)), FLAGS.input_shape)
		bb[i][3] = min(int(np.round(_y_max * y_scale)), FLAGS.input_shape)

	return bb





def create_labels(bb, classes, anchors):
	""" Creates the labels for the provided image and bounding boxes 
		Input:
			bb: array, bouding boxes of each object in the current image
			classes: list, class_id associated with the each bounding in the current image
			anchors: array, anchors for the given dataset
		Output:
			y_true: array, containing the label for the given image
	"""

	assert (classes<FLAGS.num_classes).all(), 'class_id must be less than num_classes'

	# Checking if image width and height is a multiple of 32 as YOLO has a stride of 32
	assert not FLAGS.input_shape % 32, 'Input shape must be a multiple of 32 but is {}'.format(
		FLAGS.input_shape)


	num_anchors = np.shape(anchors)[0]

	# Using default YOLOv3 settings
	num_layers = num_anchors//3 # Number of output layers
	anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [
		[3, 4, 5], [1, 2, 3]] # Which anchor is to be associated to which output layer

	# print('Bounding boxes, x1, y1, x2, y2:\n{}'.format(bb))
	# print("Number of bounding boxes: {}".format(bb.shape[0]))
	true_boxes = np.array(bb, dtype='float32')
	input_shape = np.array((FLAGS.input_shape, FLAGS.input_shape), dtype='int32')
	boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
	boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]


	# true_boxes[..., 0:2] = boxes_xy
	# true_boxes[..., 2:4] = boxes_wh
	# print('Un-normalized true_boxes, x, y, w, h:\n{}'.format(true_boxes))
	
	# Normalizing the BB
	true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
	true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

	# print('Normalized true_boxes, x, y, w, h:\n{}'.format(true_boxes))

	if (true_boxes>=1).any():
		true_boxes[true_boxes>=1] = 0.999
		# print(true_boxes)
		# exit()

	if len(true_boxes.shape) == 1:
		true_boxes = np.reshape(true_boxes, (1, true_boxes.shape[0]))

	num_boxes = true_boxes.shape[0]

	grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
	# print('Grid shapes for the input {}: {}'.format(input_shape, grid_shapes))

	y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]),
		5+FLAGS.num_classes), dtype='float32') for l in range(num_layers)]

	# print('Shape of y_true: {}'.format(np.shape(y_true)))
	# print('Shape of y_true[0]: {}\nShape of y_true[1]: {}\nShape of y_true[2]: {}'.format(
	# 	np.shape(y_true[0]), np.shape(y_true[1]), np.shape(y_true[2])))

	anchors = np.expand_dims(anchors, 0)
	anchor_maxes = anchors / 2.
	anchor_mins = -anchor_maxes
	valid_mask = boxes_wh[..., 0]>0


	wh = boxes_wh

	# Expand dimentions to apply broadcasting
	wh = np.expand_dims(wh, -2)
	box_maxes = wh / 2
	box_mins = -box_maxes

	intersect_mins = np.maximum(box_mins, anchor_mins)
	intersect_maxes = np.minimum(box_maxes, anchor_maxes)
	intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
	intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
	box_area = wh[..., 0] * wh[..., 1]
	anchor_area = anchors[..., 0] * anchors[..., 1]

	iou = intersect_area / (box_area + anchor_area - intersect_area)

	# Find best anchor for each true box
	best_anchor = np.argmax(iou, axis=-1)
	# print('Shape of true_boxes: {}\nShape of best_anchors: {}'.format(
	# 	true_boxes.shape, best_anchor.shape))

	# print('Best anchors:\n{}'.format(best_anchor))


	for t, n, in enumerate(best_anchor):
		for l in range(num_layers):
			if n in anchor_mask[l]:
				i = np.floor(true_boxes[t, 1]*grid_shapes[l][0]).astype('int32')
				j = np.floor(true_boxes[t, 0]*grid_shapes[l][1]).astype('int32')
				k = anchor_mask[l].index(n)
				# print(i, j, n, anchor_mask[l], k)
				c = classes[t].astype('int32')
				y_true[l][i, j, k, 0:4] = true_boxes[t, 0:4]
				y_true[l][i, j, k, 4] = 1
				y_true[l][i, j, k, 5+c] = 1

	return np.array(y_true)



def convert_to_example(image_data, label):
	""" Converts the values to Tensorflow TFRecord example for saving in the TFRecord file 
		Input:
			image_data: array, containing the image data read from the disk
			label: array, contains the label for the image
		Output:
			returns a Tensorflow tfrecord example
	"""
	img_raw = image_data.tostring()
	scale1 = label[0]
	scale2 = label[1]
	scale3 = label[2]
	scale1_raw = scale1.tostring()
	scale2_raw = scale2.tostring()
	scale3_raw = scale3.tostring()
	example = tf.train.Example(features=tf.train.Features(feature={
		'image_data': _bytes_feature(img_raw),
		'scale1': _bytes_feature(scale1_raw),
		'scale2': _bytes_feature(scale2_raw),
		'scale3': _bytes_feature(scale3_raw)}))
	return example



def process_tfrecord(mode, file_names, bb, classes, anchors):
	""" Makes required threds and calls further functions to execute the process of 
		making tfrecords in a multithreaded environment 
		Input:
			mode: string, specify if the tfrecords are to be made for training, validation 
				or testing
			file_names: array, containing the relative filepaths of images
			bb: array, containing bounding boxes of all the objects in an image
			classes: array, containing classes associated to every bounding box
			anchors: array, anchors for the gived dataset
			split: split done on the data for current mode
	"""

	# Checking if the passed arguments are correct
	assert len(file_names) == len(bb), 'Number of files and bouding boxes must be equal'
	assert len(bb) == len(classes), 'Number of bounding boxes and classes must be equal'

	if mode == 'train':
		num_threads = FLAGS.train_threads
		num_shards = FLAGS.train_shards

	if mode == 'val' or mode == 'test':
		num_threads = FLAGS.val_threads
		num_shards = FLAGS.val_shards

	num_anchors = np.shape(anchors)[0]

	# Getting the number of images (spacing) to be used by each thread
	spacing = np.linspace(0, len(file_names), num_threads+1).astype(np.int)
	ranges = []
	for i in range(len(spacing)-1):
		ranges.append([spacing[i], spacing[i+1]])

	print("Launching %d threads for spacings: %s" % (num_threads, ranges))

	# For coordinating all the threads
	coord = tf.train.Coordinator()

	threads = []

	# Staring all the threads for making tfrecords
	for thread_idx in range(len(ranges)):
		args = (mode, thread_idx, ranges, file_names, bb, classes, anchors)
		t = threading.Thread(target=process_tfrecord_batch, args=args)
		t.start()
		threads.append(t)


	# Wait for all threads to finish
	coord.join(threads)
	print("%s: Finished writing all %d images in dataset" %(datetime.now(), len(file_names)))



def make_tfrecord():
	""" Does some assertions and calls other functions to create tfrecords """

	# Checking if flags and shards are in correct ratio
	assert not FLAGS.train_shards % FLAGS.train_threads, ('Please \
		make the FLAGS.num_threads commensurate with FLAGS.train_shards')
	assert not FLAGS.val_shards % FLAGS.val_threads, ('Please make \
		the FLAGS.num_threads commensurate with ''FLAGS.valtest_shards')


	print('Reading {}'.format(FLAGS.anchors_path))
	anchors = read_anchors(FLAGS.anchors_path)
	num_anchors = anchors.shape[0]
	print('Number of anchors in {}: {}'.format(FLAGS.anchors_path, num_anchors))
	

	print('Reading {}'.format(FLAGS.train_annotations))
	file_path, bounding_boxes, classes = read_annotations(FLAGS.train_annotations)

	# file_path = file_path[:4]
	# bounding_boxes = bounding_boxes[:4]
	# classes = classes[:4]
	num_images = np.shape(file_path)[0]
	print('Number of images in dataset: %d' % (num_images))

	# val_split, test_split = int(0.05*num_images), int(0.01*num_images)
	val_split = int(0.1*num_images)
	train_split = num_images-val_split
	print('Splitting data with %d training images, %d validation images' 
		% (train_split, val_split)) 

	train_filenames = file_path[:train_split]
	train_bb = bounding_boxes[:train_split]
	train_classes = classes[:train_split]


	# val_filenames = file_path[train_split: train_split+val_split]
	# val_bb = bounding_boxes[train_split: train_split+val_split]
	# val_classes = classes[train_split: train_split+val_split]

	# test_filenames = file_path[train_split+val_split:]
	# test_bb = bounding_boxes[train_split+val_split:]
	# test_classes = classes[train_split+val_split:]

	val_filenames = file_path[train_split: ]
	val_bb = bounding_boxes[train_split: ]
	val_classes = classes[train_split: ]


	print('Preparing training data....')
	process_tfrecord('train', train_filenames, train_bb, train_classes, anchors)
	
	print('Done\n\nPreparing validation data....')
	process_tfrecord('val', val_filenames, val_bb, val_classes, anchors)
	
	# print('Done\n\nPreparing testing data....')
	# process_tfrecord('valtest', test_filenames, test_bb, test_classes, anchors)



if __name__ == '__main__':
	make_tfrecord()
