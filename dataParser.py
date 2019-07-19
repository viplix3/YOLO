# Importing some necessary libraries to run the program
import tensorflow as tf
import numpy as np
import os
import sys
import threading
import random
import config
import time
from datetime import datetime


# Defining some flags
tf.app.flags.DEFINE_integer('train_threads', 5, 
	'Number of threads to be used for processing training images')
tf.app.flags.DEFINE_integer('val_threads', 2, 
	'Number of threads to be used for processing validation images')
tf.app.flags.DEFINE_integer('train_shards', 10, 
	'Number of shards for training data')
tf.app.flags.DEFINE_integer('val_shards', 2, 
	'Number of shards for validation data')


FLAGS = tf.app.flags.FLAGS



class Parser:

	def __init__(self, mode, anchors_path, output_dir, num_classes, 
		input_shape, max_boxes):
		""" Initializes the object of the parser class.
			Input:
				mode: string, sets the mode to 'train' or 'val'
				anchors_path: string, path for the anchors
				output_dir: string, path for the directory where the tfrecords will be saved
				num_classes: int, number of classes in the dataset
				input_shape: int, shape of the input to the model
				max_boxes: int, maximum number of boxes to be predicted for each class
		"""
		self.input_shape = input_shape
		self.max_boxes = max_boxes
		self.mode = mode
		self.annotations_file = {'train': config.train_annotations_file, 'val': 
		config.val_annotations_file}
		# self.dataset_dir = {'train': config.train_data_file, 'val': config.val_data_file}
		self.anchors_path = anchors_path
		self.anchors = self.read_anchors()
		self.num_classes = num_classes
		self.output_dir = output_dir
		if not os.path.exists(self.output_dir):
			os.mkdir(self.output_dir)
		file_pattern = self.output_dir + self.mode + '*.tfrecord'
		self.TfrecordFile = tf.gfile.Glob(file_pattern)
		self.class_names = self.get_classes(config.classes_path)
		if len(self.TfrecordFile) == 0:
			self.make_tfrecord()
			self.TfrecordFile = tf.gfile.Glob(file_pattern)


	def _int64_feature(self, value):
		""" Converts the given input into an int64 feature that can be used in tfrecords
			Input:
				value: value to be converte into int64 feature
			Output:
				tf.train.Int64List object encoding the int64 value that can be used in tfrecords
		"""
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



	def _bytes_feature(self, value):
		""" Converts the given input into a bytes feature that can be used in tfrecords
			Input:
				value: value to be converted into bytes feature
			Output:
				tf.train.BytesList object that can be used in tfrecords
		"""
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


	def _float_feature(self, value):
		""" Converts the given input into an float feature that can be used in tfrecords
			Input:
				value: value to be converted into float feature
			Output:
				tf.train.FloatList object that can be used in tfrecords
		"""

		return tf.train.Feature(float_list=tf.train.FloatList(value=value))



	def read_anchors(self):
		""" Reads the anchors computer by k-means.py for from the provided path
			Output:
				A numpy array containing the anchors written into anchors.txt
		"""
		anchors = []
		with open(self.anchors_path, 'r') as file:
			for line in file.read().splitlines():
				w, h = line.split()
				anchor = [float(w), float(h)]
				anchors.append(anchor)

		return np.asarray(anchors)



	def get_classes(self, classes_path):
	    """ Loads the classes 
	    	Input:
	    		classes_path: path to the file containing class names
	    	Output: list containing class names
	    """
	    
	    with open(classes_path) as f:
	        class_names = f.readlines()
	    class_names = [c.strip() for c in class_names]
	    return class_names



	def read_annotations(self, file_path):
		""" Reads the image_path and annotations from train.txt
			Input:
				file_path: path to file contatining annotations
			Output:
				file_name: array, containing relative path of dataset files
				BB: array, containing Bouding Boxes coordinates for each file_name row
				class_id: class_id for each file_name row
		"""
		classes = self.class_names
		file_name = []
		BB = []
		class_id = []
		with open(file_path) as file:
			for lines in file.read().splitlines():
				line = lines.split()
				name = line[0]
				file_name.append(name)
				line = line[1::]
				_BB = []
				_class_id = []

				for i in range(len(line)):
					_BB.append(line[i].split(',')[:-1])
					_class_id.append(int(line[i].split(',')[-1]))


				# print(name, _BB, _class_id)
				# time.sleep(6)
				BB.append(np.array(_BB, dtype='float32'))
				class_id.append(np.array(_class_id, dtype='int32'))

		return np.array(file_name), np.array(BB), np.array(class_id)



	def process_tfrecord_batch(self, mode, thread_index, ranges, file_names, bb, classes):
		""" Processes images and saves tfrecords 
			Input:
				mode: string, specify if the tfrecords are to be made for training, validation 
					or testing
				thread_index: specifies the thread which is executing the function
				ranges: list, specifies the range of images the thread calling this function 
					will process
				file_names: array, containing the relative filepaths of images
				bb: array, containing bounding boxes of all the objects in an image
				classes: array, containing class_id associated to every bounding box
		"""

		if mode == 'train':
			num_threads = FLAGS.train_threads
			num_shards = FLAGS.train_shards

		if mode == 'val' or mode == 'test':
			num_threads = FLAGS.val_threads
			num_shards = FLAGS.val_shards

		num_anchors = np.shape(self.anchors)[0]

		num_shards_per_batch = int(num_shards/num_threads)
		shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1], 
			num_shards_per_batch+1).astype(int)
		num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

		counter = 0
		for s in range(num_shards_per_batch):
			shard = thread_index * num_shards_per_batch + s
			output_filename = '%s-%.5d-of-%.5d.tfrecord' % (mode, shard, num_shards)
			output_file = os.path.join(self.output_dir, output_filename)
			writer = tf.python_io.TFRecordWriter(output_file)

			shard_count = 0
			files_in_shard = np.arange(shard_ranges[s], shard_ranges[s+1], dtype=int)
			
			for i in files_in_shard:

				_filename = file_names[i]
				_classes = classes[i]
				_bb = bb[i]

				image_data = self._process_image(_filename)

				example = self.convert_to_example(_filename, image_data, _bb, _classes)
				
				writer.write(example.SerializeToString())
				shard_count += 1
				counter += 1

			
			writer.close()
			print('%s [thread %d]: Wrote %d images to %s' % (datetime.now(), thread_index, 
				shard_count, output_file))
			shard_count = 0
		print('%s [thread %d]: Wrote %d images to %d shards.' % (datetime.now(), thread_index, 
			counter, num_files_in_thread))



	def _process_image(self, filename):
		""" Read image files from disk 
			Input:
				file_name: str, relative path of the image
			Output:
				img_data: array, containing the image data
		"""
		with tf.gfile.GFile(filename, 'rb') as file:
			image_data = file.read()

		return image_data



	def preprocess_true_boxes(self, bb):
		""" Creates the labels for the provided image and bounding boxes 
			Input:
				bb: array, bouding boxes of each object in the current image
			Output:
				y_true: array, containing the label for the given image
		"""

		#assert (classes<self.num_classes).all(), 'class_id must be less than num_classes'

		# Checking if image width and height is a multiple of 32 as YOLO has a stride of 32
		assert not self.input_shape % 32, 'Input shape must be a multiple of 32 but is {}'.format(
			self.input_shape)


		num_anchors = np.shape(self.anchors)[0]

		# Using default YOLOv3 settings
		num_layers = num_anchors//config.num_anchors_per_scale # Number of output layers

		anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [
			[3, 4, 5], [0, 1, 2]] # Which anchor is to be associated to which output layer

		true_boxes = np.array(bb, dtype='float32')
		input_shape = np.array((self.input_shape, self.input_shape), dtype='int32')
		boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
		boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

		true_boxes[..., 0:2] = boxes_xy
		true_boxes[..., 2:4] = boxes_wh

		num_boxes = true_boxes.shape[0]

		grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]

		y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]),
			5+self.num_classes), dtype='float32') for l in range(num_layers)]

		anchors = np.expand_dims(self.anchors, 0)
		anchor_maxes = anchors / 2.
		anchor_mins = -anchor_maxes
		valid_mask = boxes_wh[..., 0]>0


		wh = boxes_wh[valid_mask]

		# Expand dimentions to apply broadcasting
		wh = np.expand_dims(wh, -2)
		box_maxes = wh / 2.
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

		for t, n, in enumerate(best_anchor):
			for l in range(num_layers):
				if n in anchor_mask[l]:
					i = np.floor(true_boxes[t, 0] / self.input_shape * grid_shapes[l][1]).astype('int32')
					j = np.floor(true_boxes[t, 1] / self.input_shape * grid_shapes[l][0]).astype('int32')
					k = anchor_mask[l].index(n)
					c = true_boxes[t, 4].astype('int32')
					y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
					y_true[l][j, i, k, 4:5] = 1.
					y_true[l][j, i, k, 5+c] = 1.

					# label smoothning
					# one_shot_label = y_true[l][j, i, k, 5:]
					# uniform_class_distribution = np.full(self.num_classes, 1.0/self.num_classes)
					# epsilon = 0.01
					# smooth_label = (one_shot_label * (1 - epsilon)) + (epsilon * uniform_class_distribution)
					# y_true[l][j, i, k, 5:] = smooth_label

		return y_true[0], y_true[1], y_true[2]



	def convert_to_example(self, file_name, image_data, bb, classes):
		""" Converts the values to Tensorflow TFRecord example for saving in the TFRecord file 
			Input:
				image_data: array, containing the image data read from the disk
				bb: array, containing the bounding boxes
				classes: array, containing the classes for each bounding box
			Output:
				returns a Tensorflow tfrecord example
		"""
		bb = bb.T
		classes = classes.T
		xmin = bb[0]
		ymin = bb[1]
		xmax = bb[2]
		ymax = bb[3]
		example = tf.train.Example(features=tf.train.Features(feature={
			'image/file_name': self._bytes_feature(tf.compat.as_bytes(file_name)),
			'image/encoded': self._bytes_feature(image_data),
			'image/object/bbox/xmin': self._float_feature(xmin),
			'image/object/bbox/xmax': self._float_feature(xmax),
			'image/object/bbox/ymin': self._float_feature(ymin),
			'image/object/bbox/ymax': self._float_feature(ymax),
			'image/object/bbox/label': self._float_feature(classes),
			}))
		return example



	def process_tfrecord(self, mode, file_names, bb, classes):
		""" Makes required threds and calls further functions to execute the process of 
			making tfrecords in a multithreaded environment 
			Input:
				mode: string, specify if the tfrecords are to be made for training or validation
				file_names: array, containing the relative filepaths of images
				bb: array, containing bounding boxes of all the objects in an image
				classes: array, containing classes associated to every bounding box
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

		num_anchors = np.shape(self.anchors)[0]

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
			args = (mode, thread_idx, ranges, file_names, bb, classes)
			t = threading.Thread(target=self.process_tfrecord_batch, args=args)
			t.start()
			threads.append(t)


		# Wait for all threads to finish
		coord.join(threads)
		print("%s: Finished writing all %d images in dataset" %(datetime.now(), len(file_names)))



	def make_tfrecord(self):
		""" Does some assertions and calls other functions to create tfrecords """

		# Checking if flags and shards are in correct ratio
		assert not FLAGS.train_shards % FLAGS.train_threads, ('Please \
			make the FLAGS.num_threads commensurate with FLAGS.train_shards')
		assert not FLAGS.val_shards % FLAGS.val_threads, ('Please make \
			the FLAGS.num_threads commensurate with ''FLAGS.valtest_shards')


		num_anchors = self.anchors.shape[0]
		print('Number of anchors in {}: {}'.format(self.anchors_path, num_anchors))
		

		print('Reading {}'.format(self.annotations_file[self.mode]))
		file_path, bounding_boxes, classes = self.read_annotations(self.annotations_file[self.mode])

		num_images = np.shape(file_path)[0]
		print('Number of images in dataset: %d' % (num_images))

		print('Preparing data....')
		self.process_tfrecord(self.mode, file_path, bounding_boxes, classes)
		

	def parser(self, serialized_example):
		""" Parsed the bianary serialized example
			Input:
				serialized_example, tensorflow tfrecords serialized example
			Output:
				image: tf tensor, conatines the image data
				bbox: list, containing the bounding boxes for the image
				bbox_true_19, bbox_true_38, bbox_true_76: tf tensor, containes the processed bounding boxes
		"""
		features = tf.parse_single_example(
			serialized_example,
			features = {
				'image/file_name': tf.VarLenFeature(dtype=tf.string),
				'image/encoded' : tf.FixedLenFeature([], dtype=tf.string),
				'image/object/bbox/xmin' : tf.VarLenFeature(dtype=tf.float32),
				'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
				'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
				'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
				'image/object/bbox/label': tf.VarLenFeature(dtype=tf.float32)
			}
		)
		file_name = features['image/file_name'].values
		# file_name = tf.Print(file_name, [file_name], message="file_name: ")
		image = tf.image.decode_jpeg(features['image/encoded'], channels = 3)
		image = tf.image.convert_image_dtype(image, tf.uint8)
		xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, axis=0)
		ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, axis=0)
		xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, axis=0)
		ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, axis=0)
		label = tf.expand_dims(features['image/object/bbox/label'].values, axis=0)
		bbox = tf.concat(axis=0, values=[xmin, ymin, xmax, ymax, label])
		bbox = tf.transpose(bbox, [1, 0])
		image, bbox = self.Preprocess(image, bbox)
		
		bbox_true_19, bbox_true_38, bbox_true_76 = tf.py_func(self.preprocess_true_boxes, [bbox], [tf.float32, tf.float32, tf.float32])
		return image, bbox, bbox_true_19, bbox_true_38, bbox_true_76


	def Preprocess(self, image, bbox):
		""" Resizes the image to required width and height without changing the aspect ratio,
			required prep-processing is done as well.
			Input:
				image: image for doing the pre-processing
				bbox: bounding boxes for all the iobjects in the given image
			Output:
				returns the image after doing pre-processing
		"""

		image_width, image_high = tf.cast(tf.shape(image)[1], tf.float32), tf.cast(tf.shape(image)[0], tf.float32)
		input_width = tf.cast(self.input_shape, tf.float32)
		input_high = tf.cast(self.input_shape, tf.float32)

		# Getting the new image width and height for resizing image by preserving the aspect ratio
		new_high = image_high * tf.minimum(input_width / image_width, input_high / image_high)
		new_width = image_width * tf.minimum(input_width / image_width, input_high / image_high)

		# Pixels to be added on the height and width respectively
		dx = (input_width - new_width) / 2
		dy = (input_high - new_high) / 2

		# Resizing the image
		image = tf.image.resize_images(image, [tf.cast(new_high, tf.int32), tf.cast(new_width, tf.int32)], method=tf.image.ResizeMethod.BICUBIC)
		# Padding done
		new_image = tf.image.pad_to_bounding_box(image, tf.cast(dy, tf.int32), tf.cast(dx, tf.int32), tf.cast(input_high, tf.int32), tf.cast(input_width, tf.int32))
		
		# Making the background for tapsting the image onto so that model gets the required image size
		image_ones = tf.ones_like(image)
		image_ones_padded = tf.image.pad_to_bounding_box(image_ones, tf.cast(dy, tf.int32), tf.cast(dx, tf.int32), tf.cast(input_high, tf.int32), tf.cast(input_width, tf.int32))
		
		# Making space for adding the image pixels onto the background
		image_color_padded = (1 - image_ones_padded) * 128
		image = image_color_padded + new_image

		xmin, ymin, xmax, ymax, label = tf.split(value=bbox, num_or_size_splits=5, axis=1)
		xmin = xmin * new_width / image_width + dx
		xmax = xmax * new_width / image_width + dx
		ymin = ymin * new_high / image_high + dy
		ymax = ymax * new_high / image_high + dy
		bbox = tf.concat([xmin, ymin, xmax, ymax, label], 1)
		if self.mode == 'train':
			def _flip_left_right_boxes(boxes):
				xmin, ymin, xmax, ymax, label = tf.split(value=boxes, num_or_size_splits=5, axis = 1)
				flipped_xmin = tf.subtract(input_width, xmax)
				flipped_xmax = tf.subtract(input_width, xmin)
				flipped_boxes = tf.concat([flipped_xmin, ymin, flipped_xmax, ymax, label], 1)
				return flipped_boxes
			flip_left_right = tf.greater(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1), 0.3)
			image = tf.cond(flip_left_right, lambda: tf.image.flip_left_right(image), lambda: image)
			bbox = tf.cond(flip_left_right, lambda: _flip_left_right_boxes(bbox), lambda: bbox)

			random_saturation = tf.greater(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1), 0.3)
			image = tf.cond(random_saturation, lambda: tf.image.random_saturation(image=image, lower=0.4, upper=config.sat), lambda: image)

			random_hue = tf.greater(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1), 0.3)
			image = tf.cond(random_hue, lambda: tf.image.random_hue(image=image, max_delta=config.hue), lambda: image)

			random_contrast = tf.greater(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1), 0.3)
			image = tf.cond(random_contrast, lambda: tf.image.random_contrast(image=image, lower=0.4, upper=config.cont), lambda: image)

			random_brit = tf.greater(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1), 0.3)
			image = tf.cond(random_brit, lambda: tf.image.random_brightness(image=image, max_delta=config.bri), lambda: image)

		image = image / 255.
		image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
		bbox = tf.clip_by_value(bbox, clip_value_min=0, clip_value_max=self.input_shape - 1)
		bbox = tf.cond(tf.greater(tf.shape(bbox)[0], config.max_boxes), lambda: bbox[:config.max_boxes], lambda: tf.pad(bbox, paddings=[[0, config.max_boxes - tf.shape(bbox)[0]], [0, 0]], mode = 'CONSTANT'))
		return image, bbox


	def build_dataset(self, batch_size):
		""" Builds the dataset according to the provided mode.
			Input:
				batch_size: int, batch_size to be fed into the model.
			Output:
				dataset: tf.data.Dataset object
		"""

		with tf.name_scope('data_parser/'):
			dataset = tf.data.TFRecordDataset(filenames=self.TfrecordFile)
			dataset = dataset.map(self.parser, num_parallel_calls=config.num_parallel_calls)
			if self.mode == 'train':
				dataset = dataset.repeat().shuffle(500).batch(batch_size).prefetch(batch_size)
			else:
				dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)
			return dataset
