import tensorflow as tf
import os


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


def list_tfrecord_file(folder, file_list):

    
	tfrecord_list = []
	for i in range(len(file_list)):
		current_file_abs_path = os.path.join(folder, file_list[i])		
		if current_file_abs_path.endswith(".tfrecord"):
			tfrecord_list.append(current_file_abs_path)
			#print("Found %s successfully!" % file_list[i])				
		else:
			pass
	return tfrecord_list


	
# Traverse current directory
def tfrecord_auto_traversal(folder, current_folder_filename_list):

	if current_folder_filename_list != None:
		print("%s files were found under %s folder. " % (len(current_folder_filename_list), 
			folder))
		print("Please be noted that only files ending with '*.tfrecord' will be loaded!")
		tfrecord_list = list_tfrecord_file(folder, current_folder_filename_list)
		if len(tfrecord_list) != 0:
			print("Found %d files:\n %s\n\n\n" %(len(tfrecord_list), 
				current_folder_filename_list))
		else:
			print("Cannot find any tfrecord files, please check the path.")
	return tfrecord_list



def read_tf_records(filename, image_width=416, image_height=416, num_channels=3, 
	batch_size=10):
	""" Reads tfrecords, converts the binary data to serialized data, then returns it 
		in the proper format
		Input:
			filename: string, path of the tfrecords file
			image_width: int, width of the saved images into the tfrecords
			image_height: int, height of the saved images into the tfrecords
			num_channels: int, channels of the saved images in the tfrecords
			batch_size: int, number of the images to be loaded into a single batch
		Output:
			images: uint8 tensor, images read from the tfrecords
			scale1: 4D tensor, holding the labels of the image in 32 strided scale
			scale2: 4D tensor, holding the labels of the image in 16 strided scale
			scale3: 4D tensor, holding the labels of the image in 8 strided scale
	"""

	# Features written and to be loaded from the tfrecord
	features = {'image_data': tf.FixedLenFeature([], tf.string),
				'scale1': tf.FixedLenFeature([], tf.string),
				'scale2': tf.FixedLenFeature([], tf.string),
				'scale3': tf.FixedLenFeature([], tf.string)}

	# Size of the 3 scales
	scale1_size = [batch_size, 13, 13, 3, 6] # 32 strided scale
	scale2_size = [batch_size, 26, 26, 3, 6] # 16 strided scale
	scale3_size = [batch_size, 52, 52, 3, 6] # 8 strided scale

	# Minimum number of examples to be kept into the queue for loading into the batch
	min_after_dequeue = 100
	capacity = min_after_dequeue + 8*batch_size


	for i in range(len(filename)):
		# Creating the input queue holding all the tfrecords for fetching examples
		filename_queue = tf.train.string_input_producer([filename[i]], num_epochs=None)

	# Reads the binary file and converts it into a tensorflow serialized example
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)

	# Loads the batches using the serialized example
	batch = tf.train.shuffle_batch([serialized_example], batch_size=batch_size, 
		capacity=capacity, num_threads=4, 
			min_after_dequeue=min_after_dequeue)
	parsed_example = tf.parse_example(batch, features=features)

	# Converts the banry serialized file back to the string
	image_raw = tf.decode_raw(parsed_example['image_data'], tf.uint8)


	# Reshapes the image tensor from [batch_size*image_width*image_height*num_channels] 
	# into [batch_size, image_width, image_height, num_channels]
	images = tf.reshape(image_raw, [batch_size, image_width, image_height, num_channels])

	_scale1_raw = tf.decode_raw(parsed_example['scale1'], tf.float32)
	scale1 = tf.cast(tf.reshape(_scale1_raw, scale1_size), tf.float32)

	_scale2_raw = tf.decode_raw(parsed_example['scale2'], tf.float32)
	scale2 = tf.cast(tf.reshape(_scale2_raw, scale2_size), tf.float32)

	_scale3_raw = tf.decode_raw(parsed_example['scale3'], tf.float32)
	scale3 = tf.cast(tf.reshape(_scale3_raw, scale3_size), tf.float32)


	return(images, scale1, scale2, scale3)

	# parsed_example = tf.parse_single_example(serialized_example, features=features)
	# image_height = tf.cast(parsed_example['height'], tf.int32)
	# image_width = tf.cast(parsed_example['width'], tf.int32)
	# image_raw = tf.decode_raw(parsed_example['image_data'], tf.float32)
	# images = tf.reshape(image_raw, [image_height, image_width, num_channels])

	# distorted_image = tf.random_crop(images, [530, 530, num_channels])
	# distorted_image = tf.image.random_flip_left_right(distorted_image)
	# distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
	# distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
	# distorted_image = tf.image.resize_images(distorted_image, (608, 608))
	# float_image = tf.image.per_image_standardization(distorted_image)


	# _scale1_raw = tf.decode_raw(parsed_example['scale1'], tf.float32)
	# scale1 = tf.cast(tf.reshape(_scale1_raw, scale1_size), tf.float32)

	# _scale2_raw = tf.decode_raw(parsed_example['scale2'], tf.float32)
	# scale2 = tf.cast(tf.reshape(_scale2_raw, scale2_size), tf.float32)

	# _scale3_raw = tf.decode_raw(parsed_example['scale3'], tf.float32)
	# scale3 = tf.cast(tf.reshape(_scale3_raw, scale3_size), tf.float32)

	# return tf.train.shuffle_batch(
	# 	[distorted_image, scale1, scale2, scale3],
	# 	batch_size=batch_size,
	# 	capacity=capacity,
	# 	min_after_dequeue=min_after_dequeue)


