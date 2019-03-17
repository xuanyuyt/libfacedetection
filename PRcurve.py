# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import xml.etree.cElementTree as et
import os


def computIOU(A, B):
    """
    计算两个box的IOU
    :param box1:
    :param box2:
    :return: IOU
    """
    W = min(A[2], B[2]) - max(A[0], B[0])
    H = min(A[3], B[3]) - max(A[1], B[1])
    if (W <= 0 or H <= 0):
        return 0
    SA = (A[2] - A[0]) * (A[3] - A[1])
    SB = (B[2] - B[0]) * (B[3] - B[1])
    cross = W * H
    iou = float(cross) / (SA + SB - cross)
    return iou

def readXML(xml_name, gtLabelSet):
    """
    从xml文件中读取box信息
    :param xml_name: xml文件
    :param stLabelSet:
    :return: boxes, width, height
    """
    tree = et.parse(xml_name)  # 打开xml文档
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
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        boundingBox.append([int(xmin), int(ymin), int(
            xmax), int(ymax), label, difficult])
    return boundingBox, width, height

def save_data(resultList, **kwargs):
    """
    将不同conf阈值下的TP、FP、FN结果保存
    :param resultList: 检测结果列表文件(无后缀）
    :param kwargs: gtLabelSet: ground truth 中待统计目标标签 / preLabelSet: prediction 中待统计目标标签
                    limitInPre=True 表示计算 TP/FN 时只统计preLabelSet中目标标签
    :return: 检测结果同名的mat文件，记录FP,TP,FN
    """
    resultList = os.path.join(model_root, resultList)
    gtLabelSet = kwargs.pop('gtLabelSet', ['person', 'rider', 'car'])
    preLabelSet = kwargs.pop('preLabelSet', ['person', 'rider', 'car'])
    limitInPre = kwargs.pop('limitInPre', True)
    xml_replace = kwargs.pop('xml_replace', 'Annotations')
    gt_min_limt = kwargs.pop('gt_min_limt', [320, 24])
    pre_min_limit = kwargs.pop('pre_min_limit', [320, 24])

    if limitInPre:
        PR_mat = resultList.strip().replace('.txt', '_{}_G{}P{}.mat'.format(
            preLabelSet[0], gt_min_limt[LABEL_MAP[gtLabelSet[0]]], pre_min_limit[LABEL_MAP[preLabelSet[0]]]))
    else:
        PR_mat = resultList.strip().replace('.txt', '_{}_G{}P{}_OLD.mat'.format(
            preLabelSet[0], gt_min_limt[LABEL_MAP[gtLabelSet[0]]], pre_min_limit[LABEL_MAP[preLabelSet[0]]]))

    # =========================== 打印参数信息 ============================ #
    print '# ================================================================== #'
    print ('统计模型结果：{}'.format(resultList))
    print ('a.统计GT类别：{}'.format(gtLabelSet))
    print ('b.PRE：{}'.format(preLabelSet))
    if limitInPre:
        print ('c.计算TP/FN时限制GT类别为{}'.format(preLabelSet))
    print ('d.标注版本：{}'.format(xml_replace))
    print ('e.GT大小限制：{}'.format(gt_min_limt))
    print ('e.PRE大小限制：{}'.format(pre_min_limit))
    print ('f.保存结果：{}'.format(PR_mat))

    total_pre_num = 0
    with open(resultList) as fp_pre:  # 对于每个测试图片
        for resultFile in fp_pre:
            result_datas = resultFile.strip().split('\t')
            xml_name = os.path.join(ROOTDIR,
                                    result_datas[0].replace(
                                        'JPEGImages', xml_replace) + '.xml')  # xml文件完整路径
            # print xml_name.decode("gbk")
            # ============================== 读取GT ============================== #
            if os.path.exists(xml_name):
                true_boxes, width, height = readXML(xml_name, gtLabelSet)
            else:
                print "not exit " + xml_name.decode("gbk")

            # =========================== 读取检测结果 ============================ #
            result_boxes_total = int(result_datas[1])  # 检测box数量
            detectBoxes = []
            for i in range(0, result_boxes_total, 1):  # 对每个result box
                label = result_datas[6 * i + 7]
                if not label in preLabelSet:
                    continue
                conf = float(result_datas[6 * i + 2])  # 分类置信度
                if conf < conf_thresholds[0]:
                    continue
                xmin = int(result_datas[6 * i + 3])
                ymin = int(result_datas[6 * i + 4])
                xmax = int(result_datas[6 * i + 5])
                ymax = int(result_datas[6 * i + 6])
                # =========================== 约束检测目标大小beg
                pre_width, pre_height = xmax - xmin, ymax - ymin
                limit_width = width * pre_min_limit[LABEL_MAP[label]] / resize_width
                limit_height = height * pre_min_limit[LABEL_MAP[label]] / resize_height
                if pre_width < limit_width and pre_height < limit_height:
                    continue
                # =========================== 约束检测目标大小end
                detectBoxes.append([xmin, ymin, xmax, ymax, label, conf])
            total_pre_num += len(detectBoxes)
            # =========================== 按照不同置信度统计检测情况 ============================ #
            for conf_i, conf_threshold in enumerate(conf_thresholds):
                TP = 0  # 正检
                FP = 0  # 误检
                FN = 0  # 漏检
                # ============================== FP ============================== #
                for idxP, boxP in enumerate(detectBoxes):  # 对每个result box
                    not_match = 0  # 未匹配次数
                    if boxP[5] >= conf_threshold and boxP[5] <= 1.0:  # 属于该分类阈值下的检测结果
                        for boxT in true_boxes:  # 遍历ground_truth
                            if (computIOU(boxT, boxP) < overlap_threshold):
                                not_match += 1
                        # 没有一个gt box能和result box匹配则为误检
                        if not_match == len(true_boxes):
                            # print idxP, 0
                            FP += 1

                # ============================== FN ============================== #
                for boxT in true_boxes:  # 对每个 ground_truth
                    # =========================== 约束GT目标类别beg
                    if limitInPre and not boxT[4] in preLabelSet:
                        continue
                    # =========================== 约束GT目标类别end
                    # =========================== 约束GT目标大小beg
                    gt_width, gt_height = boxT[2] - boxT[0], boxT[3] - boxT[1]
                    limit_width = width * gt_min_limt[LABEL_MAP[boxT[4]]] / resize_width
                    limit_height = height * gt_min_limt[LABEL_MAP[boxT[4]]] / resize_height
                    if gt_width < limit_width and gt_height < limit_height:
                        continue
                    # =========================== 约束GT目标大小end
                    is_match = False
                    for idxP, boxP in enumerate(detectBoxes):
                        # 如果有任意一个检测框分数够了且能和ground_truth_box 匹配上则TP+1
                        if boxP[5] >= conf_threshold and computIOU(boxT, boxP) >= overlap_threshold:
                            is_match = True
                            TP += 1  # 正确检测
                            # print idxP, 1
                            break

                    if not is_match:  # 漏检
                        FN += 1

                all_change_group[conf_i]['TP'] += TP
                all_change_group[conf_i]['FP'] += FP
                all_change_group[conf_i]['FN'] += FN

    print 'total_pre_num:{}'.format(total_pre_num)
    scipy.io.savemat(PR_mat, {'all_change_group': all_change_group})

def draw_curve(*curves):
    """
    绘制PR曲线
    :param curves: 统计结果mat文件（不限制数量）
    :return:
    """
    # 四种类别线（点）型
    ppt = ['-', '--', ':', '-.']
    ppl = ['o', '*', '+', 'x']
    for curve_i, curve_name in enumerate(curves):
        curve_name = os.path.join(model_root, curve_name)
        data = scipy.io.loadmat(curve_name)
        data = data['all_change_group'][0]
        Pos = []
        recalls = []
        precisions = []
        data_name = (curve_name.split('/')[-1])  # 图例名称
        for conf_i in range(0, len(conf_thresholds), 1):
            TP = float(data[conf_i]['TP'])
            FP = float(data[conf_i]['FP'])
            FN = float(data[conf_i]['FN'])
            P = TP + FN
            if TP == 0:
                recall = 0
                precision = 0
            else:
                recall = TP / (TP + FN)
                precision = TP / (TP + FP)
            Pos.append(P)
            recalls.append(recall)
            precisions.append(precision)
        axes.plot(recalls, precisions, ppt[model], linewidth=2, color=colors[curve_i],
                  label=[data_name, int(Pos[0])])  # 绘制每一条recall曲线
        plt.plot(recalls, precisions, ppl[model],
                 color=colors[curve_i], markersize=2)

        for conf_i, threshold in enumerate(conf_thresholds):
            plt.annotate(threshold, fontsize=6, xy=(recalls[conf_i], precisions[conf_i]),
                         xytext=(recalls[conf_i], precisions[conf_i]),
                         arrowprops=dict(
                             facecolor="w", headlength=3, headwidth=3, width=1)
                         )
    plt.legend(loc="lower left")
    # 画对角线
    # plt.plot([0.5, 1], [0.5, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.ylim([0.6, 1.0])
    # plt.xlim([0.1, 1.0])
    # plt.yticks(np.arange(0, 1.01, 0.1))  # 设置x轴刻度
    # plt.xticks(np.arange(0, 1.01, 0.1))  # 设置x轴刻度
    plt.title('Precision-Recall')
    fig.tight_layout()
    # plt.show()


# colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()
colors = ['Blue', 'Red', 'Black', 'Cyan', 'Brown', 'ForestGreen',
          'HotPink', 'Purple', 'Gold', 'Violet', 'DeepSkyBlue']
conf_thresholds = np.array(
    [ 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], dtype=np.float64)
ROOTDIR = "G:/Other_Datasets/"  # 样本根目录
model_root = "./"
overlap_threshold = 0.5


all_change_group = []  # 初始化
for j in range(0, len(conf_thresholds), 1):
    all_change_group.append({'TP': 0, 'FP': 0, 'FN': 0})

LABEL_MAP = {
    'background': 0,
    'face': 1
}
resize_width = 320
resize_height = 240

if __name__ == "__main__":
    # ================================================================== #
    #                           Draw PR curve                            #
    # ================================================================== #

    # save_data(
    #     'models/caffe/yufacedetectnet-open-v1_ALL.txt',
    #     gtLabelSet=['face'],  # 'rider', 'tricycle', 'car', 'person', 'others', 'special'
    #     preLabelSet=['face'],                         # 'rider', 'tricycle', 'car', 'person'
    #     limitInPre=True,  # 计算TP和FN是否只统计GT中在preLabelSet集合中类别
    #     xml_replace="Annotations",
    #     gt_min_limt=[240, 24],
    #     pre_min_limit=[240, 24]
    # )

    # 曲线数量+各个曲线对应的统计结果文件
    fig, axes = plt.subplots(nrows=1, figsize=(10, 8))
    plt.ion()
    model = 0
    # ================================================================================================================ #

    draw_curve(
        "models/caffe/yufacedetectnet-open-v1_ALL"
        + "_{}_G{}P{}".format('face', 24, 24),
    )
    model += 1

    # ==============================================================================================================
    plt.ioff()
    plt.grid()
    plt.show()
