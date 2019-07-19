import tensorflow as tf
from utils.utils import get_head
import config


def compute_loss(output, y_true, anchors, num_classes, input_shape, ignore_threshold=0.5, print_loss=False):
    """ Computes the custom written loss for provided output.
        Input:
            output: array, output of model for provided input image
            y_true: array, y_true label corresponding to the output produced from GT
            anchors: list, anchors for model
            num_classes: int, number of classes in the dataset
            ignore_threshold: float, threshold for considering a predicted box as True Positive
            print_loss: python boolean, flag for printing loss of the model
        Output:
            loss: computed loss
    """
    num_anchors = len(anchors) / config.num_anchors_per_scale
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_anchors==3 else [[3, 4, 5], [0, 1, 2]]

    loss_scale = []
    input_shape = [input_shape, input_shape]
    loss, loss_xy, loss_wh, loss_object, loss_no_object, loss_conf, loss_class = 0, 0, 0, 0, 0, 0, 0
    m = tf.shape(output[0])[0] # getting the batch_size for carrying out MSE
    mf = tf.cast(m, dtype=output[0].dtype)

    for l in range(len(output)):
        object_mask = y_true[l][..., 4:5]
        grid_shape = tf.shape(output[l])[1:3] # output is of shape: [batch_size, grid_x, grid_y, num_anchors_per_scale*(5+num_classes)]

        xy_offset, raw_pred, pred_xy, pred_wh = get_head(output[l], anchors[anchor_mask[l]], 
            num_classes, input_shape, calc_loss=True)
        pred_box = tf.concat([pred_xy, pred_wh], axis=-1)

        true_box = tf.boolean_mask(y_true[l][..., 0:4], tf.cast(object_mask[..., 0], dtype=tf.bool))
        true_box_xy, true_box_wh = true_box[..., 0:2], true_box[..., 2:4]
        true_box = tf.concat([true_box_xy, true_box_wh], axis=-1)

        iou = compute_iou(pred_box, true_box)
        best_iou = tf.reduce_max(iou, axis=-1)
        ignore_mask = tf.cast(best_iou < ignore_threshold, dtype=tf.float32)
        ignore_mask = tf.expand_dims(ignore_mask, axis=-1)

        # Just another way of calculating ignore_mask
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

        # Confining the range of values between 0 and 1 as we will be working with offsets w.r.t. grids
        true_xy = y_true[l][..., 0:2] / (tf.cast(input_shape[::-1] / tf.cast(grid_shape[::-1], dtype=output[l].dtype),
            dtype=output[l].dtype)) - xy_offset
        pred_xy = pred_xy / (tf.cast(input_shape[::-1] / tf.cast(grid_shape[::-1], dtype=output[l].dtype),
            dtype=output[l].dtype)) - xy_offset

        # Getting the scaling of acnhors
        true_wh = y_true[l][..., 2:4] / anchors[anchor_mask[l]]
        pred_wh = pred_wh / anchors[anchor_mask[l]]

        # for avoiding log(0) = -inf
        true_wh = tf.where(condition=tf.equal(true_wh, 0.),
                              x=tf.ones_like(true_wh), y=true_wh)
        pred_wh = tf.where(condition=tf.equal(pred_wh, 0.),
                              x=tf.ones_like(pred_wh), y=pred_wh)

        # Taking square root so that width and height errors of small and large boxes are penalised similiarly
        true_wh = tf.sqrt(true_wh)
        pred_wh = tf.sqrt(pred_wh)

        true_wh = tf.log(tf.clip_by_value(true_wh, 1e-4, 1e4))
        pred_wh = tf.log(tf.clip_by_value(pred_wh, 1e-4, 1e4))

        pred_conf = raw_pred[..., 4:5]
        true_class_probs = y_true[l][..., 5:]
        pred_class_probs = raw_pred[..., 5:]

        # box with smaller area has bigger weight
        # shape: [N, 13, 13, 3, 1]
        # print_op_gt = tf.Print(y_true[l], [tf.shape(y_true[l])], message="gt_shape: ")
        # print_op_pred = tf.Print(raw_pred, [tf.shape(raw_pred)], message="pred_shape: ")
        # with tf.control_dependencies([print_op_gt, print_op_pred]):
        box_loss_scale = 2. - (y_true[l][..., 2:3] / tf.cast(input_shape[0], tf.float32)) * (y_true[l][..., 3:4] / tf.cast(input_shape[0], tf.float32))

        # with tf.name_scope('xy_loss/'):
        xy_loss = object_mask * box_loss_scale * tf.square(true_xy - pred_xy)

        # with tf.name_scope('wh_loss/'):
        wh_loss = object_mask * box_loss_scale * tf.square(true_wh - pred_wh)

        # with tf.name_scope('confidence_loss/'):
        def calc_scale(alpha, targets, preds, gamma):
            """ Computes dynamic scaling for confidence 
                Input:
                    alpha: float, fraction of calculated scaling to be used
                    targets: tf tensor, gt labels for the batch
                    preds: tf tensor, predicted labels for the batch
                    gamma: int, power to be used for scaling
                Output:
                    Returns scaling for the required batch
            """
            return alpha * tf.pow(tf.abs(targets - tf.nn.sigmoid(preds)), gamma)

        confidence_scale = calc_scale(alpha=0.5, targets=object_mask, preds=pred_conf, gamma=2) # Calculate dynamic scaling for the model
        # confidence_scale = 1.0
        object_loss = confidence_scale * (object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf))
        no_object_loss = confidence_scale * ((1.0 - object_mask) * ignore_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf))
        confidence_loss = object_loss + no_object_loss

        # confidence_loss = confidence_scale * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf)
        
        # with tf.name_scope('class_loss/'):
        class_scale = calc_scale(alpha=0.5, targets=true_class_probs, preds=pred_class_probs, gamma=2)
        # class_scale = 1.0
        class_loss = class_scale * object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_class_probs, logits=pred_class_probs)

        # class_loss = object_mask * tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_class_probs, logits=pred_class_probs)
        # class_loss = class_scale * object_mask * tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_class_probs, logits=pred_class_probs)

        # Taking mean of all the losses over the batch size
        xy_loss = tf.reduce_sum(xy_loss) / mf
        wh_loss = tf.reduce_sum(wh_loss) / mf
        object_loss = tf.reduce_sum(object_loss) / mf
        no_object_loss = tf.reduce_sum(no_object_loss) / mf
        confidence_loss = tf.reduce_sum(confidence_loss) / mf
        class_loss = tf.reduce_sum(class_loss) / mf

        # Adding losses for all the 3 scales
        loss_xy += xy_loss
        loss_wh += wh_loss
        loss_object += object_loss
        loss_no_object += no_object_loss
        loss_conf += confidence_loss
        loss_class += class_loss
        loss += (xy_loss + wh_loss + confidence_loss + class_loss)
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
 
