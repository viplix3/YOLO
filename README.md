# YOLOv3 Tensorflow
------

### This is an implementation of YOLOv3 in tensorflow. Although this can work for other variants of YOLO as well with minute changes.
### [YAD2K](https://github.com/allanzelener/YAD2K) and [darknet](https://github.com/pjreddie/darknet) has been a great help.

## TODO
------
- [x] YOLOv3-darknet to tensorflow conversion (inference only)
- [x] Inference script
- [x] Input pipeline
- [x] Speed evaluation
- [ ] mAP evaluation
- [ ] Training on VOC

**Suggestions and pull requests are most welcome**

### Running YOLOv3-608 model on a 1080Ti I am getting around 19FPS
### Running YOLOv3-608 model on a 1050Ti I am getting around 6FPS

## Some Results
------
![alt-text](/results/images/RWovpT.jpeg)("YOLO-tensorflow results")

## How to run the model you ask?
------

Make a virtual environment and install all the dependencies

I am using virtualenv.

Run the following commands

```shell
virtualenv env -p python3
source env\bin\activate
```
```python
pip install -r requirements.txt
python inference.py path_to_the_image_directory path_for_saving_the_results --darknet_model 1`
```
The above command runs the pretrained model.

## Training the model
------

Now this will be some work..

### 1. Data prepration

Try to find an object detection dataset which is having annotations in VOC format as my annotation parsing script works for only that. (Sorry COCO lovers)

I suggest [raccoon-dataset](https://github.com/experiencor/raccoon_dataset) provided by [experiencor](https://github.com/experiencor). (Who is having a great implementation of YOLOv3 in keras.)

Make a directory and put your dataset with two folders, one holding images and the other holding the corresponding annotations.
**There must be a one-to-one correspondence by file name between images and annotations.**

I suggest the following structure.

```
+ dataset
|
|___
	+ dataset_name
	|
	|___
	|	+ images
	|	|
	|	|___ + all the image files
	|___
		+ annotations
		|
		|___ + all the annotation files
```

### 2. Telling the model what the new classes are

Make a dataset-name_classes.txt file inside the model_data folder, having one class of the dataset in one line.
A sample file named sample_classes.txt has been provided for reference.


### 3. Updating configuration file

The configuration file is a python .py file containing variables, updation of which is pretty self explainatory by reading the comments provided.


### 4. Generating the anchors for your dataset (recommended but not mandatory)

If you are doing this step, update the variable anchors_path in your config.py file to point to the location where you want to save the newly generated anchors.

Run the following command

```python
python k-means.py
```

### 5. Actually running the training. (Finally, yes I know the feel!!)

Running the following command will do some darkmagic as done in darknet (pun intended) and will start training the model.

```python
python train.py
```
You might get an error while training for the first time, I am still working to resolve it. (Suggestions are welcome, I am still a noob.)

Run the command again and you are good to go.

### 6. Making detections using the trained weights

```python
python inference.py path_to_the_image_directory path_for_saving_the_results`
```