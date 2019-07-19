import cv2
import config
import numpy as np

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

def main():
	anchors = read_anchors(config.anchors_path)
	image = np.zeros((config.input_shape, config.input_shape))
	mid_x, mid_y = config.input_shape//2, config.input_shape//2
	for i in range(anchors.shape[0]):
		x1, y1, x2, y2 = int(mid_x - anchors[i][0]//2), int(mid_y - anchors[i][1]//2), int(mid_x + anchors[i][0]//2), int(mid_y + anchors[i][1]//2)
		print(x1, y1, x2, y2)
		cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
	cv2.imwrite('./anchors.png', image)


if __name__ == '__main__':
	main() 
