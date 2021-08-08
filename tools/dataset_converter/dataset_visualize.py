#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import numpy as np
import cv2
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_classes, get_dataset, get_colors, draw_boxes


def dataset_visualize(annotation_file, classes_path):

    annotation_lines = get_dataset(annotation_file, shuffle=False)
    # get class names and count class item number
    class_names = get_classes(classes_path)
    colors = get_colors(len(class_names))

    pbar = tqdm(total=len(annotation_lines), desc='Visualize dataset')
    for annotation_line in annotation_lines:
        pbar.update(1)
        line = annotation_line.split()
        image = cv2.imread(line[0])
        boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        classes = boxes[:, -1]
        boxes = boxes[:, :-1]
        scores = np.array([1.0]*len(classes))

        image = draw_boxes(image, boxes, classes, scores, class_names, colors, show_score=False)

        cv2.namedWindow("Image", 0)
        cv2.imshow("Image", image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    pbar.close()


def main():
    parser = argparse.ArgumentParser(description='visualize dataset')
    parser.add_argument('--annotation_file', required=True, type=str, help='txt annotation file path')
    parser.add_argument('--classes_path', type=str, required=True, help='path to class definitions')

    args = parser.parse_args()

    dataset_visualize(args.annotation_file, args.classes_path)


if __name__ == '__main__':
    main()
