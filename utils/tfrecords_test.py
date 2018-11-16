import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.patches as patches
from read_tfrecord import *

tfrecords = tfrecord_auto_traversal('./tfrecords', os.listdir('./tfrecords'))

image_tensor, label1_tensor, label2_tensor, label3_tensor = read_tf_records(tfrecords)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True



with open('./model_data/raccoon_classes.txt') as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]


with tf.Session(config=config) as sess:

	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	image, label1, label2, label3 = sess.run([image_tensor, label1_tensor, label2_tensor, 
		label3_tensor])

	print('Image tensor shape: {}\nLabels Shape:\n1: {}\n2: {}\n3: {}'.format(image.shape, 
		label1.shape, label2.shape, label3.shape))


	for i in range(len(image)):
		r1, r2, r3 = [], [], []
		fix, ax = plt.subplots()
		ax.imshow(image[i])
		for x in range(13):
			for y in range(13):
				for z in range(3):
					if label1[i, x, y, z, 4] == 1:
						r1.append(label1[i, x, y, z, 0:4])
						conf = label1[i, x, y, z, 4]
						clas = np.argmax(label1[i, x, y, z, 5:])
		r1 = np.array(r1)
		r1 *= 416
		print('r1 shape: {}\nr1: {}\n\n'.format(r1.shape, r1))
		if r1.shape[0]:
			for q in range(r1.shape[0]):
				x, y = r1[q][0], r1[q][1]
				w, h = r1[q][2], r1[q][3]
				rect = patches.Rectangle((int(x - w /2), int(y - h / 2)), (w), (h), linewidth=2, 
					edgecolor='r', facecolor='none')
				ax.add_patch(rect)
				ax.text(x, y, class_names[clas], horizontalalignment='left', 
					verticalalignment='bottom', color='b')

		for x in range(26):
			for y in range(26):
				for z in range(3):
					if label2[i, x, y, z, 4] == 1:
						r2.append(label2[i, x, y, z, 0:4])
						conf = label2[i, x, y, z, 4]
						clas = np.argmax(label2[i, x, y, z, 5:])
		r2 = np.array(r2)
		r2 *= 416
		print('r2 shape: {}\nr2: {}\n\n'.format(r2.shape, r2))
		if r2.shape[0]:
			for q in range(r2.shape[0]):
				x, y = r2[q][0], r2[q][1]
				w, h = r2[q][2], r2[q][3]
				rect = patches.Rectangle((int(x - w /2), int(y - h / 2)), (w), (h), linewidth=2, 
					edgecolor='r', facecolor='none')
				ax.add_patch(rect)
				ax.text(x, y, class_names[clas], horizontalalignment='left', 
					verticalalignment='bottom', color='b')


		for x in range(52):
			for y in range(52):
				for z in range(3):
					if label3[i, x, y, z, 4] == 1:
						r3.append(label3[i, x, y, z, 0:4])
						conf = label3[i, x, y, z, 4]
						clas = np.argmax(label3[i, x, y, z, 5:])
		r3 = np.array(r3)
		r3 *= 416
		print('r3 shape: {}\nr3: {}\n\n'.format(r3.shape, r3))
		if r3.shape[0]:
			for q in range(r3.shape[0]):
				x, y = r3[q][0], r3[q][1]
				w, h = r3[q][2], r3[q][3]
				rect = patches.Rectangle((int(x - w /2), int(y - h / 2)), (w), (h), linewidth=2, 
					edgecolor='r', facecolor='none')
				ax.add_patch(rect)
				ax.text(x, y, class_names[clas], horizontalalignment='left', 
					verticalalignment='bottom', color='b')


		plt.show()
		print('\n\n\n')

	coord.request_stop()
	coord.join(threads)
