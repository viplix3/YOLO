""" A program which performs k-means clustering to get the the dimentions of anchor boxes """
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import config
from utils.utils import resize_image

file_path = config.train_annotations_file


def iou(boxes, clusters):
	""" Calculates Intersection over Union between the provided boxes and cluster centroids
		Input:
			boxes: Bounding boxes
			clusters: cluster centroids
		Output:
			IoU between boxes and cluster centroids
	"""
	n = boxes.shape[0]
	k = np.shape(clusters)[0]

	box_area = boxes[:, 0] * boxes[:, 1] # Area = width * height
	# Repeating the area for every cluster as we need to calculate IoU with every cluster
	box_area = box_area.repeat(k)
	box_area = np.reshape(box_area, (n, k))

	cluster_area = clusters[:, 0] * clusters[:, 1]
	cluster_area = np.tile(cluster_area, [1, n])
	cluster_area = np.reshape(cluster_area, (n, k))


	box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
	cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
	min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

	box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
	cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
	min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
	inter_area = np.multiply(min_w_matrix, min_h_matrix)

	result = inter_area / (box_area + cluster_area - inter_area)
	return result


def avg_iou(boxes, clusters):
	""" Calculates average IoU between the GT boxes and clusters 
		Input:
			boxes: array, having width and height of all the GT boxes
		Output:
			Returns numpy array of average IoU with all the clusters
	"""
	return np.mean([np.max(iou(boxes, clusters), axis=1)])


def kmeans(boxes, k):
	""" Executes k-means clustering on the rovided width and height of boxes with IoU as
		distance metric.
		Input:
			boxes: numpy array containing with and height of all the boxes
		Output:
			clusters after convergence
	"""
	num_boxes = boxes.shape[0]
	distances = np.empty((num_boxes, k))
	last_cluster = np.zeros((num_boxes, ))

	# Initializing the clusters
	np.random.seed()
	clusters = boxes[np.random.choice(num_boxes, k, replace=False)]

	# Optimizarion loop
	while True:

		distances = 1 - iou(boxes, clusters)
		mean_distance = np.mean(distances)
		sys.stdout.write('\r>> Mean loss: %f' % (mean_distance))
		sys.stdout.flush()

		current_nearest = np.argmin(distances, axis=1)
		if(last_cluster == current_nearest).all():
			break # The model is converged
		for cluster in range(k):
			clusters[cluster] = np.mean(boxes[current_nearest == cluster], axis=0)

		last_cluster = current_nearest
	return clusters


def dump_results(data):
	""" Writes the anchors after running k-means clustering onto the disk for further usage
		Input:
			data: array, containing the data for anchor boxes
	"""
	f = open("./anchors.txt", 'w')
	row = np.shape(data)[0]
	for i in range(row):
		x_y = "%d %d\n" % (data[i][0], data[i][1])
		f.write(x_y)
	f.close()


def get_boxes(file_path):
	""" Extracts the bounding boxes from the coco train.txt file 
		Input:
			file_path: path of train.txt made from coco annotations
		Output:
			numpy array containing all the bouding boxes width and height
	"""
	with open(file_path, 'r') as f:
		dataSet = []
		for line in f:
			infos = line.split(' ')
			length = len(infos)
			sys.stdout.write('\r>> Reading image: %s' % (infos[0].split('/')[-1]))
			sys.stdout.flush()
			img = cv2.imread(infos[0])
			image_width, image_height = img.shape[1], img.shape[0]
			scale = np.minimum(config.input_shape / image_width, config.input_shape / image_height)
			new_width, new_height = image_width * scale, image_height * scale
			dx = (config.input_shape - new_width) / 2
			dy = (config.input_shape - new_height) / 2
			# In every line of train.txt the values are stored as:
			# [relative_image_path, x1, y1, x2, y2, class_id]
			for i in range(1, length):
				xmin, xmax = int(infos[i].split(',')[0]), int(infos[i].split(',')[2])
				ymin, ymax = int(infos[i].split(',')[1]), int(infos[i].split(',')[3])
				xmin = int(xmin * new_width/image_width + dx)
				xmax = int(xmax * new_width/image_width + dx)
				ymin = int(ymin * new_height/image_height + dy)
				ymax = int(ymax * new_height/image_height + dy)
				width = xmax - xmin
				height = ymax - ymin
				if (width == 0) or (height == 0):
					continue
				dataSet.append([width, height])
	result = np.array(dataSet)
	return result


def get_clusters(num_clusters, file_path):
	""" Calls all the required functions to run k-means and get good anchor boxes 
		Input:
			num_clusters: number of clusters
			file_path: path of train.txt containing parsed annotations 
		Output:
			Returns avg_accuracy of computer anchor box over the whole dataset
	"""
	all_boxes = get_boxes(file_path)
	print('\n')
	result = kmeans(all_boxes, num_clusters)
	result = result[np.lexsort(result.T[0, None])]
	dump_results(result)
	print("\n\n{} anchors:\n{}".format(num_clusters, result))
	avg_acc = avg_iou(all_boxes, result)*100
	print("Average accuracy: {:.2f}%".format(avg_acc))

	return avg_acc


if __name__ == '__main__':

	min_cluster, max_cluster = config.num_anchors, config.num_anchors + 1
	clusters = np.arange(min_cluster, max_cluster, dtype=int)
	avg_accuracy = []
	for i in clusters:
		avg_acc = get_clusters(i, file_path)
		avg_accuracy.append(avg_acc)

	if max_cluster - min_cluster > 1:
		plt.plot(clusters, avg_accuracy)
		plt.xlabel('Number of Clusters')
		plt.ylabel('Average Accuracy')
		plt.savefig('./cluster.png')
