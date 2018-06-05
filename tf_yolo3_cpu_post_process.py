#!/usr/bin/env python
# coding=utf-8

import argparse

import colorsys
import os
import random
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from yolo3.utils import letterbox_image
from yolo3_cpu_post_process import yolo_eval


def parse_args():
    parser = argparse.ArgumentParser(description="tensorflow yolov3 inference demo")
    parser.add_argument("--model_path", "-m", required=True, help="required, path to the freezed tensorflow yolov3 model")
    parser.add_argument("--anchor_path", "-a", required=True, help="required, path to the file contains yolov3 anchors")
    parser.add_argument("--label_path", "-l", required=True, help="required, path to the text file contains all class names")
    parser.add_argument("--score_thresh", required=False, type=float, default=0.5,
                        help="threshold for the class score, default to 0.5")
    parser.add_argument("--nms_iou_thresh", required=False, type=float, default=0.45,
                        help="iou threshold for the nms post-process, default to 0.45")

    parser.add_argument("--input_image_path", "-i", help="path to the test image", default=None)

    group = parser.add_argument_group()
    group.add_argument("--input_dir", "-d", help="path to a directory contains some test images", default=None)
    group.add_argument("--output_dir", "-o", help="directory for the detecting results", default=None)

    return parser.parse_args()


def load_graph(frozen_graph_path, name_prefix=None):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        # We load the graph_def in the default graph
        # with tf.Graph().as_default() as graph:
        graph = tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=name_prefix,
            producer_op_list=None
        )

    return graph


class TFYOLO(object):
    def __init__(self, model_path, anchor_path, label_path, score_thresh=0.5, nms_iou_thresh=0.45):
        self._model_path = os.path.expanduser(model_path)
        self._anchor_path = anchor_path
        self._label_path = label_path
        self._score_thresh = score_thresh
        self._nms_iou_thresh = nms_iou_thresh

        self._class_names = self._get_class()
        self._anchors = self._get_anchors()
        self._model_image_size = (416, 416)  # fixed size or (None, None), hw

        self._sess = tf.Session()

        name_prefix = "yolov3"
        load_graph(self._model_path, name_prefix)

        graph = tf.get_default_graph()
        self._input_tensor = graph.get_tensor_by_name(name_prefix + '/input_1:0')
        self._output_tensor_list = [
            graph.get_tensor_by_name(name_prefix + '/conv2d_59/BiasAdd:0'),
            graph.get_tensor_by_name(name_prefix + '/conv2d_67/BiasAdd:0'),
            graph.get_tensor_by_name(name_prefix + '/conv2d_75/BiasAdd:0'),
        ]

        print('{} model, anchors, and classes loaded.'.format(self._model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self._class_names), 1., 1.)
                      for x in range(len(self._class_names))]
        self._colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self._colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self._colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self._colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

    def _get_class(self):
        classes_path = os.path.expanduser(self._label_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self._anchor_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def detect_image(self, image):
        start = timer()

        start1 = timer()
        if self._model_image_size != (None, None):
            assert self._model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self._model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self._model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print("")
        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        print("preprocess: %fs" % (timer() - start1))

        start1 = timer()
        yolo_outputs = self._sess.run(self._output_tensor_list,
                                      feed_dict={
                                          self._input_tensor: image_data
                                      })
        print("model inference: %fs" % (timer() - start1))

        start1 = timer()
        out_boxes, out_scores, out_classes = yolo_eval(yolo_outputs, self._anchors,
                                                       len(self._class_names), [image.size[1], image.size[0]],
                                                       score_threshold=self._score_thresh,
                                                       iou_threshold=self._nms_iou_thresh)
        print("cpu post process: %fs" % (timer() - start1))

        print("total time: %fs" % (timer() - start))

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 400

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self._class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for j in range(thickness):
                draw.rectangle(
                    [left + j, top + j, right - j, bottom - j],
                    outline=self._colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self._colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        return image

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._sess.close()
        print("tf session closed")

        return False

    def close_session(self):
        self._sess.close()


if __name__ == '__main__':
    args = parse_args()

    with TFYOLO(args.model_path, args.anchor_path, args.label_path, args.score_thresh,
                args.nms_iou_thresh) as yolo_detector:
        if args.input_image_path is not None and os.path.isfile(args.input_image_path):
            test_image = Image.open(args.input_image_path)
            result_image = yolo_detector.detect_image(test_image)
            result_image.show()

        if args.input_dir is not None and os.path.isdir(args.input_dir):
            output_dir = args.output_dir
            if output_dir is None:
                output_dir = "yolov3_results"
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            image_sufix_list = ["jpg", "jpeg", "png"]
            for name in os.listdir(args.input_dir):
                suffix = name.rsplit(".", 1)[-1]
                if suffix in image_sufix_list:
                    test_image = Image.open(os.path.join(args.input_dir, name))
                    result_image = yolo_detector.detect_image(test_image)
                    out_path = os.path.join(output_dir, name)
                    result_image.save(out_path)
