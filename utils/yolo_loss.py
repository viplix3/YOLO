import tensorflow as tf
from utils.utils import yolo_head
import config


def compute_loss(output, y_true, anchors, num_classes, ignore_threshold=0.5, object_scale=1, print_loss=False):
    """ Computes the custom written YOLO loss for provided output.
    	Input:
    		output: array, output of YOLO for provided input image
    		y_true: array, y_true label corresponding to the output produced from GT
    		anchors: list, anchors for YOLO
    		num_classes: int, number of classes in the dataset
    		ignore_threshold: float, threshold for considering a predicted box as True Positive
    	Output:
    		loss: computed loss
    """
    num_anchors = len(anchors)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_anchors==9 else [[3, 4, 5], [0, 1, 2]]

    loss_scale = []
    input_shape = [config.input_shape, config.input_shape]
    loss, loss_xy, loss_wh, loss_object, loss_no_object, loss_conf, loss_class = 0, 0, 0, 0, 0, 0, 0
    m = tf.shape(output[0])[0] # getting the batch_size
    mf = tf.cast(m, dtype=output[0].dtype)

    for l in range(len(output)):
        object_mask = y_true[l][..., 4:5]
        grid_shape = tf.shape(output[l])[1:3] # output is of shape: [batch_size, grid_x, grid_y, num_anchors_per_scale*(5+num_classes)]

        xy_offset, raw_pred, pred_xy, pred_wh = yolo_head(output[l], anchors[anchor_mask[l]], 
            num_classes, input_shape, calc_loss=True)
        pred_box = tf.concat([pred_xy, pred_wh], axis=-1)

        true_box = tf.boolean_mask(y_true[l][..., 0:4], tf.cast(object_mask[..., 0], dtype=tf.bool))
        true_box_xy, true_box_wh = true_box[..., 0:2], true_box[..., 2:4]
        true_box = tf.concat([true_box_xy, true_box_wh], axis=-1)

        iou = compute_iou(pred_box, true_box)
        best_iou = tf.reduce_max(iou, axis=-1)
        ignore_mask = tf.cast(best_iou < ignore_threshold, dtype=tf.float32)
        ignore_mask = tf.expand_dims(ignore_mask, axis=-1)


        # ignore_mask = tf.TensorArray(y_true[0].dtype, size=1, dynamic_size=True)
        # object_mask_bool = tf.cast(object_mask, dtype=tf.bool)

        # def loop_body(b, ignore_mask):
        #     true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
        #     iou = compute_iou(pred_box[b], true_box)
        #     best_iou = tf.reduce_max(iou, axis=-1)
        #     ignore_mask = ignore_mask.write(b, tf.cast(best_iou < ignore_threshold, 
        #         true_box.dtype))
        #     return b+1, ignore_mask

        # _, ignore_mask = tf.while_loop(lambda b, ignore_mask: b < m, loop_body, 
        #     [0, ignore_mask])

        # ignore_mask = ignore_mask.stack()
        # ignore_mask = tf.expand_dims(ignore_mask, axis=-1)
        

        true_xy = y_true[l][..., 0:2] / (tf.cast(input_shape[::-1] / tf.cast(grid_shape[::-1], dtype=output[l].dtype),
            dtype=output[l].dtype)) - xy_offset
        pred_xy = pred_xy / (tf.cast(input_shape[::-1] / tf.cast(grid_shape[::-1], dtype=output[l].dtype),
            dtype=output[l].dtype)) - xy_offset

        # print_op = tf.Print(true_xy, [tf.shape(true_xy)], message="xy_offset: ")
        # true_xy = tf.Print(true_xy, [tf.reduce_min(true_xy), tf.reduce_max(true_xy)], message="true_xy: ")
        # pred_xy = tf.Print(pred_xy, [tf.reduce_min(pred_xy), tf.reduce_max(pred_xy)], message="pred_xy: ")

        true_wh = y_true[l][..., 2:4] / anchors[anchor_mask[l]]
        pred_wh = pred_wh / anchors[anchor_mask[l]]

        # for avoiding log(0) = -inf
        true_wh = tf.where(condition=tf.equal(true_wh, 0.),
                              x=tf.ones_like(true_wh), y=true_wh)
        pred_wh = tf.where(condition=tf.equal(pred_wh, 0.),
                              x=tf.ones_like(pred_wh), y=pred_wh)

        true_wh = tf.sqrt(true_wh)
        pred_wh = tf.sqrt(pred_wh)

        true_wh = tf.log(tf.clip_by_value(true_wh, 1e-4, 1e4))
        pred_wh = tf.log(tf.clip_by_value(pred_wh, 1e-4, 1e4))

        # true_wh = true_wh * object_mask
        # true_wh = tf.Print(true_wh, [tf.reduce_min(true_wh), tf.reduce_max(true_wh)], message="true_wh: ")
        # pred_wh = object_mask * pred_wh
        # pred_wh = tf.Print(pred_wh, [tf.reduce_min(pred_wh), tf.reduce_max(pred_wh)], message="pred_wh: ")


        pred_conf = raw_pred[..., 4:5]
        true_class_probs = y_true[l][..., 5:]
        pred_class_probs = raw_pred[..., 5:]


        """ Computing some statistics """
        # ignore_score_threshold = 0.5
        # detect_mask = tf.to_float(tf.nn.sigmoid(pred_conf) * object_mask >= ignore_score_threshold)
        # class_mask = tf.to_float(tf.equal(tf.argmax(tf.nn.sigmoid(pred_class_probs)), tf.argmax(true_class_probs)))
        # avg_iou = tf.reduce_sum(iou) / (tf.reduce_sum(object_mask) + 1e-6)
        # recall_50 = tf.reduce_sum(tf.to_float(iou >= 0.50) * detect_mask * class_mask) / (tf.reduce_sum(object_mask) + 1e-6)
        # recall_75 = tf.reduce_sum(tf.to_float(iou >= 0.75) * detect_mask * class_mask) / (tf.reduce_sum(object_mask) + 1e-6)

        # iou_avg += avg_iou
        # recall_50_avg += recall_50
        # recall_75_avg += recall_75

        # pred_conf = tf.Print(pred_conf, [tf.reduce_min(pred_conf), tf.reduce_max(pred_conf)], message="pred_conf: ")
        # pred_class_probs = tf.Print(pred_class_probs, [tf.reduce_min(pred_class_probs), tf.reduce_max(pred_class_probs)], message="pred_class_probs: ")

        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # shape: [N, 13, 13, 3, 1]
        box_loss_scale = 2. - (y_true[l][..., 2:3] / input_shape[1]) * (y_true[l][..., 3:4] / tf.cast(input_shape[0], tf.float32))
        
        #### HUBER LOSS ####
        # delta_xy = 2.0
        # delta_wh = 2.0
        
        # err_xy = tf.subtract(true_xy, pred_xy)
        # xy_loss = object_mask * box_loss_scale * tf.where(condition=tf.less(tf.abs(err_xy), delta_xy), 
        #     x=0.5 * tf.square(err_xy), y=0.5 * tf.square(delta_xy) + delta_xy * tf.subtract(tf.abs(err_xy), delta_xy))

        # err_wh = tf.subtract(true_wh, pred_wh)
        # wh_loss = object_mask * box_loss_scale * tf.where(condition=tf.less(tf.abs(err_wh), delta_wh), 
        #     x=0.5 * tf.square(err_wh), y=0.5 * tf.square(delta_wh) + delta_wh * tf.subtract(tf.abs(err_wh), delta_wh))

        xy_loss = object_mask * box_loss_scale * tf.square(true_xy - pred_xy)

        wh_loss = object_mask * box_loss_scale * tf.square(true_wh - pred_wh)

        object_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf)
        no_object_loss = (1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf) * ignore_mask
        confidence_loss = object_loss + no_object_loss

        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_class_probs, logits=pred_class_probs)

        xy_loss = tf.reduce_sum(xy_loss) / mf
        wh_loss = tf.reduce_sum(wh_loss) / mf
        object_loss = tf.reduce_sum(object_loss) / mf
        no_object_loss = tf.reduce_sum(no_object_loss) / mf
        confidence_loss = tf.reduce_sum(confidence_loss) / mf
        class_loss = tf.reduce_sum(class_loss) / mf

        loss_xy += xy_loss
        loss_wh += wh_loss
        loss_object += object_loss
        loss_no_object += no_object_loss
        loss_conf += confidence_loss
        loss_class += class_loss
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        loss_scale.append(xy_loss + wh_loss + confidence_loss + class_loss)

        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, 
                tf.reduce_sum(ignore_mask)], message='loss: ')

    return loss_scale, loss, loss_xy, loss_wh, loss_object, loss_no_object, loss_conf, loss_class


def compute_iou(box1, box2):
    """ Computes IoU between two boxes.
    	Input:
    		box1: list, parameters for box1
    		box2: list, parameters for box 2
    	Output:
    		iou: float, iou between box1 and box2
    """
    box1 = tf.expand_dims(box1, -2)
    box1_xy = box1[..., 0:2]
    box1_wh = box1[..., 2:4]
    box1_mins = box1_xy - box1_wh / 2.
    box1_maxs = box1_xy + box1_wh / 2.

    box2 = tf.expand_dims(box2, 0)
    box2_xy = box2[..., 0:2]
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