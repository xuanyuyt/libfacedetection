# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xml.etree.cElementTree as et
import caffe
import cv2
import argparse
import os

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2


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

def readXML(xml_name, gtLabelSet):
    """
	从xml文件中读取box信息
    :param xml_name: xml文件
    :param stLabelSet:
    :return: boxes, width, height
    """
    tree = et.parse(xml_name) #打开xml文档
    # 得到文档元素对象
    root = tree.getroot()
    size = root.find('size')  # 找到root节点下的size节点
    width = int(size.find('width').text)  # 子节点下节点width的值
    height = int(size.find('height').text)  # 子节点下节点height的值

    boundingBox = []
    for object in root.findall('object'):  # 找到root节点下的所有object节点
        label = object.find('name').text
        difficult = int(object.find('difficult').text)
        if not label in gtLabelSet:
            continue
        bndbox = object.find('bndbox')  # 子节点下属性bndbox的值
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
        boundingBox.append([int(xmin), int(ymin), int(xmax), int(ymax), label, difficult])
    return boundingBox, width, height

def computIOU(A, B):
    W = min(A[2], B[2]) - max(A[0], B[0])
    H = min(A[3], B[3]) - max(A[1], B[1])
    if (W <= 0 or H <= 0):
        return 0
    SA = (A[2] - A[0]) * (A[3] - A[1])
    SB = (B[2] - B[0]) * (B[3] - B[1])
    cross = W * H
    iou = float(cross) / (SA + SB - cross)
    return iou

def main(args):
    """
    press G to next image
    press R to pre mage
    press I to any index you want
    :param args:
    :return: FN white
    """
    print main.__doc__
    model_root = unicode(args.model_root, 'utf-8').encode('gbk')
    file_root = unicode(args.file_root, 'utf-8').encode('gbk')
    model_def = os.path.join(model_root, args.model_def)
    model_weights = os.path.join(model_root, args.model_weights)

    file_list = os.path.join(model_root, args.file_list)
    all_file_list = []
    with open(file_list, 'rb') as fp:
        for oneFile in fp:
            img_name = os.path.join(file_root, oneFile.strip().split('\t')[0])
            # xml_name = os.path.join(file_root, oneFile.strip().split('\t')[1])
            xml_name = img_name.replace('JPEGImages', args.xml_version).replace('.jpg', '.xml')
            all_file_list.append([img_name, xml_name])

    # load PASCAL VOC labels
    labelmap_file = os.path.join(model_root, args.labelmap_file)
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)
    batch_size = 1


    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights4
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104,117,123])) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    # set net to batch size of 1
    resize_width = args.input_size[0]
    resize_height = args.input_size[1]
    net.blobs['data'].reshape(1,3,resize_height,resize_width)

    conf_ths = args.conf_ths
    colors = [(0, 0, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]

    TPs = 0  # 正检
    FPs = 0  # 误检
    FNs = 0  # 漏检
    cv2.namedWindow('Detect', 0)
    cv2.namedWindow('GroundTruth', 0)
    img_index = 1
    while(img_index):
        if img_index > len(all_file_list):
            break
        img_name = all_file_list[img_index-1][0]
        xml_name = all_file_list[img_index-1][1]
        # ============================== 读取GT & IMG ============================== #
        if os.path.exists(xml_name):
            # 所有的gtLabelSet ground truth boxes
            true_boxes, width, height = readXML(xml_name, args.select_type_set)
        else:
            print "not exit " + xml_name.decode("gbk")
        print img_name.decode("gbk"), width, height

        cv_image = cv2.imread(img_name)
        cv_image2 = cv_image.copy()
        # ==============================
        # image = cv2.resize(cv_image.copy(), (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        # image = (image.copy() - [104, 117, 123])
        # transformed_image = image.transpose(2, 0, 1)
        # ==============================
        image = caffe.io.load_image(img_name)
        transformed_image = transformer.preprocess('data', image)
        # ==============================
        net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = net.forward()['detection_out']
        # -----------------------------------------------------------------------
        # feature = net.blobs['detection_out'].data[0]
        # output = open(u'D:/Backup/桌面/COM/detection_out_Python.txt', 'w')
        # for c in xrange(feature.shape[0]):
        #     for h in xrange(feature.shape[1]):
        #         for w in xrange(feature.shape[2]):
        #             output.write(str(feature[c, h, w]))
        #             output.write('\n')
        # output.close()
        # -----------------------------------------------------------------------
        # Parse the outputs.
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_ths[int(det_label[i])]]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        TP = 0  # 正检
        FP = 0  # 误检
        FN = 0  # 漏检
        detectBoxes = []
        for idx in xrange(top_conf.shape[0]):  # 对每个检测到的目标
            not_match = 0
            xmin = int(round(top_xmin[idx] * width))
            ymin = int(round(top_ymin[idx] * height))
            xmax = int(round(top_xmax[idx] * width))
            ymax = int(round(top_ymax[idx] * height))
            score = top_conf[idx]
            label = top_labels[idx]
            # =========================== 约束检测目标大小beg
            pre_width, pre_height = xmax - xmin, ymax - ymin
            limit_width = width * args.pre_min_limit[LABEL_MAP[label]] / resize_width
            limit_height = height * args.pre_min_limit[LABEL_MAP[label]] / resize_height
            if pre_width < limit_width and pre_height < limit_height:
                continue
            # =========================== 约束检测目标大小end
            # 变成了负样本, 或者第一阶段分数不够
            if label == 'background' or score < conf_ths[LABEL_MAP[label]]:
                continue
            detectBoxes.append([xmin, ymin, xmax, ymax, label, score])

        # ============================== comp & draw ============================== #
        # 绘制误检
        for boxP in detectBoxes:
            xmin, ymin, xmax, ymax, label, score = boxP
            # cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            not_match = 0
            limit_gt_box_num = 0
            for boxT in true_boxes:
                if boxT[4] in [label, 'others']:
                    limit_gt_box_num += 1
                    if (computIOU(boxT, boxP) < 0.5):
                        not_match += 1  # 未匹配次数
            if not_match == limit_gt_box_num:
                FP += 1
                size = max((xmax - xmin), (ymax - ymin))
                display_txt = '%s: %d' % (label, size)
                color = colors[LABEL_MAP[label]]
                cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(cv_image, display_txt, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

        # 绘制GT和漏检
        for boxT in true_boxes:
            xmin, ymin, xmax, ymax, label, diff = boxT
            size = max((xmax - xmin), (ymax - ymin))
            color = (255, 255, 255) # 白色
            display_txt = '%s: %d' % (label, size)
            cv2.rectangle(cv_image2, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(cv_image2, display_txt, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
            if label == 'others':
                continue
            # =========================== 约束GT目标大小beg
            gt_width, gt_height = xmax - xmin, ymax - ymin
            limit_width = width * args.gt_min_limt[LABEL_MAP[label]] / 1920
            limit_height = height * args.gt_min_limt[LABEL_MAP[label]] / 1080
            if gt_width < limit_width and gt_height < limit_height:
                continue
            # =========================== 约束GT目标大小end
            is_match = False
            for boxP in detectBoxes:
                # 如果有一个检测框,分数够了且分类正确
                if boxP[5] >= conf_ths[LABEL_MAP[boxP[4]]] and boxP[4] == boxT[4]:
                    # 如果有任意一个检测框能和ground_truth_box 匹配上则TP+1
                    if (computIOU(boxT, boxP) > 0.5):
                        is_match = True
                        TP += 1  # 正确检测
                        break
            if not is_match:  # 漏检
                FN += 1
                cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(cv_image, display_txt, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

        display_txt2 = 'TP: %i FP: %i FN: %i'%(TP, FP, FN)
        cv2.putText(cv_image, display_txt2, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        cv2.imshow('Detect', cv_image)
        cv2.imshow('GroundTruth', cv_image2)

        TPs += TP
        FPs += FP
        FNs += FN
        key = cv2.waitKey(0) & 0xFF
        if key == ord('g'):
            img_index += 1
        elif key == ord('r'):
            img_index -= 1
        elif key == ord('i'):
            try:
                img_index = int(raw_input("Enter your input: "))
            except Exception as e:
                print e
        else:
            img_index += 1


    print 'TPs: %i FPs: %i FNs: %i'%(TPs, FPs, FNs)

LABEL_MAP = {
    'background':0,
    'face': 1,
     }
if __name__ == "__main__":
    # ================================================================== #
    #                       Show Detection result                        #
    # ================================================================== #
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_root', type=str,
                        default='',
                        help='path for model')
    parser.add_argument('--file_root', type=str,
                        default='G:/Other_Datasets',
                        help='path for src ini file')
    parser.add_argument('--file_list', type=str, default='ALL.txt',
                        help='ini file list')
    parser.add_argument('--model_def', type=str,
                        default='models/caffe/yufacedetectnet-open-v1.prototxt',
                        help='sub path prototxt file')
    parser.add_argument('--model_weights', type=str,
                        default='models/caffe/yufacedetectnet-open-v1.caffemodel',
                        help='sub path weight file')
    parser.add_argument('--input_size', type=int, nargs='+',
                        default=[320, 240],
                        help='Examples: --input_size 320 240')
    parser.add_argument('--labelmap_file', type=str,
                        default='labelmap_Face.prototxt',
                        help='label map file')
    parser.add_argument('--select_type_set', type=str, nargs='+',
                        default=['face'],
                        help='Examples: --select_type_set face')
    parser.add_argument('--conf_ths', type=float, nargs='+',
                        default=[1.0, 0.5],
                        help='Examples: --conf_ths 1.0 0.5')
    parser.add_argument('--pre_min_limit', type=int, nargs='+',
                        default=[240, 24],
                        help='Examples: --max_limt 256 64')
    parser.add_argument('--gt_min_limt', type=int, nargs='+',
                        default=[240, 24],
                        help='Examples: --max_limt 256 64')
    parser.add_argument('--xml_version', type=str, default='Annotations',
                        help='Annotations')
    args = parser.parse_args()
    main(args)
