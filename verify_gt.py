import cv2
import numpy as np
import config
import os


def gt_verification(file_path):
	""" Reads the image_path and annotations from train.txt
		Input:
			file_path: path to file contatining annotations
		Output:
			file_name: array, containing relative path of dataset files
			BB: array, containing Bouding Boxes coordinates for each file_name row
			class_id: class_id for each file_name row
	"""
	class_id = []
	file_name = []
	BB = []
	num_images_to_verify = 10
	idx_index_to_verify = np.random.randint(low=0, high=config.train_num, size=num_images_to_verify)
	idx = 0
	with open(file_path) as file:
		for lines in file.read().splitlines():
			line = lines.split()
			name = line[0]
			file_name.append(name)
			line = line[1::]
			_BB = []
			_class_id = []

			if idx in idx_index_to_verify:
				image = cv2.imread(name)

				for i in range(len(line)):
					_BB.append(line[i].split(',')[:-1])
					_class_id.append(int(line[i].split(',')[-1]))

				for i in range(len(_BB)):
					cv2.rectangle(image, (int(_BB[i][0]), int(_BB[i][1])), (int(_BB[i][2]), int(_BB[i][3])), (0, 0, 255), 1)
					image = image.copy()

				cv2.imwrite(config.gt_verification_folder+name.split('/')[-1], image)
				BB.append(np.array(_BB, dtype='float32'))
				class_id.append(np.array(_class_id, dtype='int32'))
			idx += 1

if __name__ == '__main__':
	os.makedirs(config.gt_verification_folder, exist_ok=True)
	gt_verification(config.train_annotations_file)