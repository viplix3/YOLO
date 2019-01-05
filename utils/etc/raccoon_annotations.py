import xml.etree.ElementTree as ET
import os

classes = ["raccoon"]


def convert_annotation(image_id, list_file):
    in_file = open('./dataset/raccoon/annotations/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


images = os.listdir('./dataset/raccoon/images/')
num_images = len(images)
num_train_images = num_images * 0.8
num_val_images = num_images - num_train_images
train_file = open('train.txt', 'w')
val_file = open('val.txt', 'w')
list_file = train_file
num = 0

cwd = os.getcwd()
for image in images:
    num += 1
    list_file.write(os.path.join(os.path.join(cwd,'dataset/raccoon/images'), image))
    convert_annotation(image.split('.')[0], list_file)
    list_file.write('\n')
    if num == num_train_images:
        list_file.close()
        list_file = val_file
list_file.close()

