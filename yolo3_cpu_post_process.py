#!/usr/bin/env python
# coding=utf8

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def cpu_nms(boxes, box_scores, iou_thresh, max_kept_boxes):
    y_min = boxes[:, 0]
    x_min = boxes[:, 1]
    y_max = boxes[:, 2]
    x_max = boxes[:, 3]
    box_areas = (x_max - x_min) * (y_max - y_min)
    score_order = box_scores.argsort()[::-1]

    keep_indexes = []
    while score_order.size > 0 and len(keep_indexes) < max_kept_boxes:
        current_max_score_index = score_order[0]
        keep_indexes.append(current_max_score_index)
        intersections_lefts = np.maximum(x_min[current_max_score_index], x_min[score_order[1:]])
        intersections_bottoms = np.maximum(y_min[current_max_score_index], y_min[score_order[1:]])
        intersections_rights = np.minimum(x_max[current_max_score_index], x_max[score_order[1:]])
        intersections_tops = np.minimum(y_max[current_max_score_index], y_max[score_order[1:]])

        intersections_widths = np.maximum(0.0, intersections_rights - intersections_lefts + 1)
        intersections_heights = np.maximum(0.0, intersections_tops - intersections_bottoms + 1)

        intersections_areas = intersections_widths * intersections_heights
        iou = intersections_areas / (box_areas[current_max_score_index]
                                     + box_areas[score_order[1:]] - intersections_areas)
        suppressed_boxes_index = np.where(iou <= iou_thresh)[0]
        score_order = score_order[suppressed_boxes_index + 1]
    return keep_indexes


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              score_threshold,
              iou_threshold,
              max_boxes=20):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting

    out_shape = yolo_outputs[0].shape
    input_shape = np.array(out_shape[1:3]) * 32

    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = \
            yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)

    boxes = np.concatenate(tuple(boxes), axis=0)
    box_scores = np.concatenate(tuple(box_scores), axis=0)

    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        class_box_scores = box_scores[:, c]
        selected = class_box_scores >= score_threshold
        selected_class_boxes = boxes[selected]
        if len(selected_class_boxes) < 1:
            continue
        selected_class_box_scores = class_box_scores[selected]

        nms_indices = cpu_nms(selected_class_boxes, selected_class_box_scores, iou_threshold, max_boxes)
        if len(nms_indices) < 1:
            continue

        selected_class_boxes = selected_class_boxes[nms_indices]
        selected_class_box_scores = selected_class_box_scores[nms_indices]
        classes = np.ones_like(selected_class_box_scores, dtype=np.int32) * c

        boxes_.append(selected_class_boxes)
        scores_.append(selected_class_box_scores)
        classes_.append(classes)

    boxes_ = np.concatenate(tuple(boxes_), axis=0)
    scores_ = np.concatenate(tuple(scores_), axis=0)
    classes_ = np.concatenate(tuple(classes_), axis=0)

    return boxes_, scores_, classes_


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
                                                                anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = np.reshape(boxes, (-1, 4))
    box_scores = box_confidence * box_class_probs
    box_scores = np.reshape(box_scores, (-1, num_classes))
    return boxes, box_scores


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors = np.reshape(anchors, (1, 1, 1, num_anchors, 2))

    grid_shape = np.array(feats.shape[1:3])  # height, width
    grid_y = np.tile(np.reshape(np.arange(0, grid_shape[0]), (-1, 1, 1, 1)), [1, grid_shape[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(0, grid_shape[1]), (1, -1, 1, 1)), [grid_shape[0], 1, 1, 1])

    grid = np.concatenate((grid_x, grid_y), axis=-1)
    grid = grid.astype(feats.dtype)

    feats = np.reshape(feats, (-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5))

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (sigmoid(feats[..., :2]) + grid) / grid_shape[::-1].astype(feats.dtype)
    box_wh = np.exp(feats[..., 2:4]) * anchors / input_shape[::-1].astype(feats.dtype)
    box_confidence = sigmoid(feats[..., 4:5])
    box_class_probs = sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    ## Get corrected boxes
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = input_shape.astype(box_yx.dtype)
    image_shape = np.array(image_shape).astype(box_yx.dtype)
    new_shape = np.round(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate((
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ), axis=-1)

    # Scale boxes back to original image shape.
    boxes *= np.concatenate((image_shape, image_shape), axis=-1)

    return boxes
