import tensorflow as tf
from utils.utils import yolo_head
import keras.backend as K


# def compute_loss(output, y_true, anchors, num_classes, ignore_threshold=0.5, print_loss=False):
#     """ Computes the custom written YOLO loss for provided output.
#     	Input:
#     		output: array, output of YOLO for provided input image
#     		y_true: array, y_true label corresponding to the output produced from GT
#     		anchors: list, anchors for YOLO
#     		num_classes: int, number of classes in the dataset
#     		ignore_threshold: float, threshold for considering a predicted box as True Positive
#     	Output:
#     		loss: computed loss
#     """
#     num_anchors = len(anchors)
#     anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_anchors==3 else [[3, 4, 5], [0, 1, 2]]

#     input_shape = tf.cast(tf.shape(output[0])[1:3] * 32, dtype=y_true[0].dtype)
#     grid_shapes = [tf.cast(tf.shape(output[l])[1:3], dtype=y_true[0].dtype) for l in range(
#         len(output))]
#     loss = 0
#     m = tf.shape(output[0])[0] # getting the batch_size
#     mf = tf.cast(m, dtype=output[0].dtype)

#     for l in range(len(output)):
#         object_mask = y_true[l][..., 4:5]
#         true_class_probs = y_true[l][..., 5:]

#         grid, raw_pred, pred_xy, pred_wh = yolo_head(output[l], anchors[anchor_mask[l]], 
#             num_classes, input_shape, calc_loss=True)
#         pred_box = tf.concat([pred_xy, pred_wh], axis=-1)

#         # Darknet raw boxes to calculate loss
#         raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
#         raw_true_wh = tf.log(tf.where(tf.equal(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * 
#             input_shape[::-1], 0), tf.ones_like(y_true[l][..., 2:4]), 
#             y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1]))


#         box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]
#         # Fnd ignore mask, iterate over each of batch
#         ignore_mask = tf.TensorArray(y_true[0].dtype, size=1, dynamic_size=True)
#         object_mask_bool = tf.cast(object_mask, dtype=tf.bool)

#         def loop_body(b, ignore_mask):
#             true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
#             iou = box_IoU(pred_box[b], true_box)
#             best_iou = tf.reduce_sum(iou, axis=-1)
#             ignore_mask = ignore_mask.write(b, tf.cast(best_iou < ignore_threshold, 
#                 true_box.dtype))
#             return b+1, ignore_mask

#         _, ignore_mask = tf.while_loop(lambda b, ignore_mask: b < m, loop_body, 
#             [0, ignore_mask])
#         ignore_mask = ignore_mask.stack()
#         ignore_mask = tf.expand_dims(ignore_mask, axis=-1)

#         xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(labels=raw_true_xy, 
#             logits=raw_pred[..., 0:2])

#         wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - raw_pred[..., 2:4])

#         confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, 
#             logits=raw_pred[..., 4:5]) + (1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=raw_pred[..., 4:5]) * ignore_mask

#         class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_class_probs, logits=raw_pred[..., 5:])

#         xy_loss = tf.reduce_sum(xy_loss) / mf
#         wh_loss = tf.reduce_sum(wh_loss) / mf
#         confdence_loss = tf.reduce_sum(confidence_loss) / mf
#         class_loss = tf.reduce_sum(class_loss) / mf
#         loss += xy_loss + wh_loss + confidence_loss + class_loss

#         if print_loss:
#             loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, 
#                 tf.reduce_sum(ignore_mask)], message='loss: ')

#         return tf.reduce_sum(loss)



def compute_loss(yolo_outputs, y_true, anchors, num_classes, ignore_thresh=0.5, print_loss=False):
    """ Computes the custom written YOLO loss for provided output.
    	Input:
    		yolo_output: list of tensor, output of YOLO for provided input image
    		y_true: list of tensor, y_true label corresponding to the output produced from GT
    		anchors: array, anchors for YOLO
    		num_classes: int, number of classes in the dataset
    		ignore_threshold: float, threshold for considering a predicted box as True Positive
    		print_loss: boolean, weather to print loss for each iteration, useful for debugging
    	Output:
    		loss: computed loss
    """
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_layers = len(yolo_outputs)
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))


    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])


        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_IoU(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        with tf.name_scope('xy_loss'):
	        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
	                                                                       from_logits=True)
        with tf.name_scope('wh_loss'):
        	wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        with tf.name_scope('conf_loss'):
	        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
	                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
	                                                                    from_logits=True) * ignore_mask
        with tf.name_scope('class_loss'):
        	class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        with tf.name_scope('total_loss'):
	        xy_loss = K.sum(xy_loss) / mf
	        wh_loss = K.sum(wh_loss) / mf
	        confidence_loss = K.sum(confidence_loss) / mf
	        class_loss = K.sum(class_loss) / mf
	        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)],
                            message='loss: ')
    return loss


def box_IoU(box1, box2):
    """ Computes IoU between two boxes.
    	Input:
    		box1: list, parameters for box1
    		box2: list, parameters for box 2
    	Output:
    		iou: float, iou between box1 and box2
    """
    box1 = tf.expand_dims(box1, -2)
    box1_xy = box1[..., :2]
    box1_wh = box1[..., 2:4]
    box1_mins = box1_xy - box1_wh / 2.
    box1_maxs = box1_xy + box1_wh / 2.

    box2 = tf.expand_dims(box2, 0)
    box2_xy = box2[..., :2]
    box2_wh = box2[..., 2:4]
    box2_mins = box2_xy - box2_wh / 2.
    box2_maxs = box2_xy + box2_wh / 2.

    intersect_mins = tf.maximum(box1_mins, box2_mins)
    intersect_maxs = tf.minimum(box1_maxs, box2_maxs)
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box1_area = box1_wh[..., 0] * box1_wh[..., 1]
    box2_area = box2_wh[..., 0] * box2_wh[..., 1]
    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou