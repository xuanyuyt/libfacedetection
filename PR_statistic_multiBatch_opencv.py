# -*- coding: utf-8 -*-
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import caffe
import cv2
import os
import numpy as np
import argparse

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def save_detection(args):
    """
    为绘制不用conf阈值下的P-R曲线做准备
    :param args:
    :return: image_name total_detect [score xmin ymin xmax ymax strlabel] ...
    """
    # ============================== print info ============================== #
    print save_detection.__doc__
    print 'use ssd model: {}'.format(args.model_weights)
    # if args.use_stage2:
    #     print 'use stage2 model: {}'.format(args.stage2_model_weights)
    print 'test {} image list'.format(args.file_list)
    print 'use {} conf ths'.format(conf_map)


    model_root = unicode(args.model_root, 'utf-8').encode('gbk')
    file_root = unicode(args.file_root, 'utf-8').encode('gbk')

    # load PASCAL VOC labels
    LABELMAP_FILE = os.path.join(model_root, args.labelmap_file)
    file = open(LABELMAP_FILE, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    # ============================== SSD prepare ============================== #
    model_def = os.path.join(model_root, args.model_def)
    model_weights = os.path.join(model_root, args.model_weights)
    net = caffe.Net(model_def,  # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)

    resize_width = args.input_size[0]
    resize_height = args.input_size[1]
    net.blobs['data'].reshape(args.batch_size, 3, resize_height, resize_width)
    mean_values = np.array([104.0, 117.0, 123.0])

    save_file = model_weights.strip().split('.caffemodel')[
        0] + '_{}'.format(args.file_list)

    if os.path.exists(save_file):
        os.remove(save_file)

    iterate = 0
    batch_index = 0
    imageNameList = []
    imageShapeList = []
    for imgFile in open(os.path.join(model_root, args.file_list)).readlines():  # 对于每个测试图片
        if (iterate == args.batch_size):
            iterate = 0
            batch_index = batch_index + 1
            # Forward pass.
            detections = net.forward()['detection_out']
            # Parse the outputs.
            result_index = detections[0, 0, :, 0]  # 所有目标所在一个batch中图片的序号（从0开始）
            det_label = detections[0, 0, :, 1]    #
            det_conf = detections[0, 0, :, 2]
            det_xmin = detections[0, 0, :, 3]
            det_ymin = detections[0, 0, :, 4]
            det_xmax = detections[0, 0, :, 5]
            det_ymax = detections[0, 0, :, 6]
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.1]
            result_index = result_index[top_indices]
            det_label = det_label[top_indices]
            det_conf = det_conf[top_indices]
            det_xmin = det_xmin[top_indices]
            det_ymin = det_ymin[top_indices]
            det_xmax = det_xmax[top_indices]
            det_ymax = det_ymax[top_indices]
            output = open(save_file, 'a')
            for index in range(0, args.batch_size):
                # Get detections with confidence higher than 0.6.
                top_indices = [i for i, part_index in enumerate(
                    result_index) if int(part_index) == index]
                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                top_labels = get_labelname(labelmap, top_label_indices)
                top_xmin = det_xmin[top_indices]
                top_ymin = det_ymin[top_indices]
                top_xmax = det_xmax[top_indices]
                top_ymax = det_ymax[top_indices]

                # Write result
                output.write(imageNameList[index])
                output.write('\t')
                output.write(str(top_conf.shape[0]))
                if top_conf.shape[0] < 1:
                    print imageNameList[index].decode("gbk"), ' no any detection'
                for i in xrange(top_conf.shape[0]):  # 对每个检测到的目标
                    xmin = int(round(top_xmin[i] * imageShapeList[index][1]))
                    ymin = int(round(top_ymin[i] * imageShapeList[index][0]))
                    xmax = int(round(top_xmax[i] * imageShapeList[index][1]))
                    ymax = int(round(top_ymax[i] * imageShapeList[index][0]))
                    label_name = top_labels[i]
                    score = top_conf[i]
                    if score < conf_map[label_name]:
                        continue
                    output.write('\t')
                    output.write(str(score))  # 2
                    output.write('\t')
                    output.write(str(xmin))  # 3
                    output.write('\t')
                    output.write(str(ymin))  # 4
                    output.write('\t')
                    output.write(str(xmax))  # 5
                    output.write('\t')
                    output.write(str(ymax))  # 6
                    output.write('\t')
                    output.write(label_name)  # 7
                output.write('\n')
            output.close()
            print '@_@ have extracted ', batch_index * args.batch_size, ' images '
            imageNameList = []
            imageShapeList = []
        # ================================================================== #
        img_name = os.path.join(file_root, imgFile.strip().split('\t')[0])
        # xml_name = os.path.join(file_root, imgFile.strip().split('\t')[1])
        imageNameList.append(imgFile.strip().split('.jpg')[0])
        tmp_cv_image = cv2.imread(img_name)
        tmp_height, tmp_width, channel = tmp_cv_image.shape
        imageShapeList.append([tmp_height, tmp_width])
        # ==============================
        image = cv2.resize(tmp_cv_image.copy(), (resize_width,
                                                 resize_height), interpolation=cv2.INTER_LINEAR)
        image = (image.copy() - mean_values)
        transformed_image = image.transpose(2, 0, 1)
        net.blobs['data'].data[iterate, :, :, :] = transformed_image
        iterate = iterate + 1

    # for the rest images less than batch_size
    # here i equals the rest images
    print '@_@  extracted the rest', iterate, ' images '
    # ================================================================== #
    # Forward pass.
    detections = net.forward()['detection_out']
    # Parse the outputs.
    result_index = detections[0, 0, :, 0]
    det_label = detections[0, 0, :, 1]
    det_conf = detections[0, 0, :, 2]
    det_xmin = detections[0, 0, :, 3]
    det_ymin = detections[0, 0, :, 4]
    det_xmax = detections[0, 0, :, 5]
    det_ymax = detections[0, 0, :, 6]
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.1]
    result_index = result_index[top_indices]
    det_label = det_label[top_indices]
    det_conf = det_conf[top_indices]
    det_xmin = det_xmin[top_indices]
    det_ymin = det_ymin[top_indices]
    det_xmax = det_xmax[top_indices]
    det_ymax = det_ymax[top_indices]
    output = open(save_file, 'a')
    for index in range(0, iterate):
        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, part_index in enumerate(
            result_index) if int(part_index) == index]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        # Write result
        output.write(imageNameList[index])
        output.write('\t')
        output.write(str(top_conf.shape[0]))
        if top_conf.shape[0] < 1:
            print imageNameList[index].decode("gbk")
        for i in xrange(top_conf.shape[0]):  # 对每个检测到的目标
            xmin = int(round(top_xmin[i] * imageShapeList[index][1]))
            ymin = int(round(top_ymin[i] * imageShapeList[index][0]))
            xmax = int(round(top_xmax[i] * imageShapeList[index][1]))
            ymax = int(round(top_ymax[i] * imageShapeList[index][0]))
            label_name = top_labels[i]
            score = top_conf[i]
            if score < conf_map[label_name]:
                continue
            output.write('\t')
            output.write(str(score))  # 2
            output.write('\t')
            output.write(str(xmin))  # 3
            output.write('\t')
            output.write(str(ymin))  # 4
            output.write('\t')
            output.write(str(xmax))  # 5
            output.write('\t')
            output.write(str(ymax))  # 6
            output.write('\t')
            output.write(label_name)  # 7
        output.write('\n')
    output.close()

    print '@_@  extracted ', batch_index*args.batch_size + iterate, ' images '
    print '.Done'


caffe.set_device(0)
caffe.set_mode_gpu()
conf_map = {
    'background':1.0,
    'face': 0.1
     }
if __name__ == "__main__":
    # ================================================================== #
    #                          保存检测结果文件                          #
    # ================================================================== #
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_root', type=str,
                        default='',
                        help='path for ssd model')
    parser.add_argument('--model_def', type=str,
                        default='models/caffe/yufacedetectnet-open-v1.prototxt',
                        help='SSD sub path prototxt file')
    parser.add_argument('--model_weights', type=str,
                        default='models/caffe/yufacedetectnet-open-v1.caffemodel',
                        help='SSD sub path weight file')
    parser.add_argument('--input_size', type=int, nargs='+',
                        default=[320, 240],
                        help='Examples: --input_size 320 240')
    parser.add_argument('--file_root', type=str,
                        default='G:/Other_Datasets',
                        help='root for src img file')
    parser.add_argument('--file_list', type=str,
                        default='ALL.txt',
                        help='img file list') # Path_Imgs2.txt
    parser.add_argument('--labelmap_file', type=str,
                        default='labelmap_Face.prototxt',
                        help='label map file')
    parser.add_argument('--select_type_set', type=str, nargs='+',
                        default=['face'],
                        help='Examples: --select_type_set face')
    parser.add_argument('--batch_size', type=int,
                        default=100,
                        help='Examples: --batch_size 100')
    args = parser.parse_args()
    save_detection(args)