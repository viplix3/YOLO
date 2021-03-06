num_parallel_calls = 10
input_shape = 320 # input shape of the model
max_boxes = 51 # max number of boxes to be kept of a class during non max suppression
# for data augmentation
jitter = 0.3 
hue = 0.1
sat = 1.5
cont = 0.8
bri = 0.1

# decay parameters
use_warm_up = False # for warm up training so that the model training goes smoothly without gradient explosion
burn_in_epochs = 1 # number of iterations to be used for burn-in
norm_decay = 0.9 # batch_norm momentum
weight_decay = 0.0005 # weight decay for l2-regularization of convolution kernels
norm_epsilon = 1e-3 # for avoiding division by zero error during batch normalization
pre_train = True # weather to use pre-trained darknet 53 feature extractor during training
feature_extractor_conv_count = 52 # number of convolutional layers in the feature extractor
num_anchors = 9 # number of anchors
num_anchors_per_scale = 3
num_scales = 3
num_classes = 80 # number of classes in the dataset
training = True # training flag
ignore_thresh = 0.5 # IoU threshold to consider a box as true positive, during calculating loss
init_learning_rate = 1e-3 # initial learning rate for carrying out the burn in process
learning_rate = 1e-3 # learning rate for training the model
learning_rate_lower_bound = 1e-5 # lower bound for learning rate
momentum = 0.9 # momentum for the optimizer
train_batch_size = 16 # batch size to be used during training
subdivisions = 4 # for splitting the training data into this many minibtches for processing
val_batch_size = 16 # batch size to be used during validation
train_num = 160
val_num = 40
Epoch = 50 # number of epochs for training
score_threshold = 0.6 # score threshold for prediction
nms_threshold = 0.3 # nms threshold for prediction
warm_up_lr_scheduler = 'polynomial' # learning rate scheduler to be used during burn-in (linear, exponential, polynomial)
lr_scheduler = "linear" # learning rate scheduler to be used after burn-in (linear, exponential, polynomial)
gpu_num = "0" # gpu bus id to be used for all the processes
logs_dir = './logs/' # path for saving the training/validation logs
dataset_dir = '/home/viplix3/Documents/GitHub/raccoon_dataset/' # base path in which the dataset has been kept
gt_verification_folder = './gt_verification/'
model_dir = './converted/' # path for saving the model
anchors_path = './model_data/yolo_anchors_pre-trained.txt' # path to the anchors file
yolov3_cfg_path = './darknet_data/yolov3.cfg' # cfg file for the model
yolov3_weights_path = './darknet_data/yolov3.weights' # path of weights file for the model if the pre-trained model is to be used for training
darknet53_weights_path = './darknet_data/darknet53.weights' # path of weights file for the darknet feature extractor
anchors_path = './anchors.txt' # path to the anchors file
classes_path = './model_data/coco_classes.txt' # path for the text file containing the classes of the dataset
train_annotations_file = './train.txt' # path for the text file containing the training image annotations
val_annotations_file = './val.txt' # path for text file containing the validation image annotations
output_dir = './tfrecords/' # path for saving the tfrecords
model_export_path = './protobuf_model/YOLOv3.pb' # path for saving the protobuf model for production purposes
