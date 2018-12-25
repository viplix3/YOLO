import xml.etree.ElementTree as ET
import os
import config


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


def convert_annotation(ann, list_file):
    in_file = ann
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


def main():
    classes = get_classes(config.classes_path)
    img_base_path = sorted(os.path.join(config.dataset_dir, 'images'))
    ann_base_path = sorted(os.path.join(config.dataset_dir, 'annotations'))
    images = os.listdir(os.path.join(config.dataset_dir, 'images'))
    ann = os.listdir(os.path.join(config.dataset_dir, 'annotations'))
    train_file = open('train.txt', 'w')
    val_file = open('val.txt', 'w')
    img_num = 0
    list_file = train_file


    for image, ann in zip(images, ann):
        img_num += 1
        list_file.write(os.path.join(img_base_path, image))
        convert_annotation(os.path.join(ann_base_path, ann), list_file)
        list_file.write('\n')
        if img_num == config.train_num:
            list_file.close()
            list_file = val_file
    list_file.close()


if __name__ == '__main__':
    main()
