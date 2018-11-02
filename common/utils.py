# coding: utf-8

import os
from os.path import join, exists
import time
import cv2
import numpy as np
from .cnns import getCNNs


def logger(msg):
    """
        log message
    """
    now = time.ctime()
    print("[%s] %s" % (now, msg))

def createDir(p):
    if not os.path.exists(p):
        os.mkdir(p)

def shuffle_in_unison_scary(a, b):
    '''
    洗牌，要保证图像和标签一一对应
    :param a:图像
    :param b:标签
    :return:
    '''
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    # 同样的随机种子
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def drawLandmark(img, bbox, landmark):
    cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 2, (0,255,0), -1)
    return img

def getDataFromTxt(txt, with_landmark=True):
    '''
    从数据标注txt里面读取数据并整理好
        Generate data from txt file
    :param txt: 文件路径
    :param with_landmark:
    :return: [(img_path, bbox, landmark)]
            bbox: [left, right, top, bottom]人脸框
            landmark: [(x1, y1), (x2, y2), ...]左眼睛、右眼睛、鼻子、左嘴角、右嘴角对应坐标
    '''
    dirname = os.path.dirname(txt)
    with open(txt, 'r') as fd:
        lines = fd.readlines()

    result = []
    for line in lines:
        line = line.strip()
        components = line.split(' ')
        img_path = os.path.join(dirname, components[0].replace('\\', '/')) # file path
        # bounding box, (left, right, top, bottom)
        bbox = (components[1], components[2], components[3], components[4])
        bbox = [int(_) for _ in bbox]
        # landmark
        if not with_landmark:
            result.append((img_path, BBox(bbox)))
            continue
        landmark = np.zeros((5, 2))
        for index in range(0, 5):
            # 左眼睛、右眼睛、鼻子、左嘴角、右嘴角对应坐标
            rv = (float(components[5+2*index]), float(components[5+2*index+1]))
            landmark[index] = rv
        for index, one in enumerate(landmark):
            # 把人脸关键点相对于图片的位置转换为相对于人脸框的位置
            # 这样可以缩小训练和测试的范围大小，可以大大的提高速度和准确率
            # 除以宽高消除人脸框宽高带来的影响
            rv = ((one[0]-bbox[0])/(bbox[1]-bbox[0]), (one[1]-bbox[2])/(bbox[3]-bbox[2]))
            landmark[index] = rv
        result.append((img_path, BBox(bbox), landmark))
    return result

def getPatch(img, bbox, point, padding):
    """
        Get a patch iamge around the given point in bbox with padding
        point: relative_point in [0, 1] in bbox
    """
    point_x = bbox.x + point[0] * bbox.w
    point_y = bbox.y + point[1] * bbox.h
    patch_left = int(point_x - bbox.w * padding)
    patch_right = int(point_x + bbox.w * padding)
    patch_top = int(point_y - bbox.h * padding)
    patch_bottom = int(point_y + bbox.h * padding)
    patch = img[patch_top: patch_bottom+1, patch_left: patch_right+1]
    patch_bbox = BBox([patch_left, patch_right, patch_top, patch_bottom])
    return patch, patch_bbox


def processImage(imgs):
    """
        process images before feeding to CNNs
        imgs: N x 1 x W x H
    """
    imgs = imgs.astype(np.float32)
    for i, img in enumerate(imgs):
        m = img.mean()
        s = img.std()
        imgs[i] = (img - m) / s
    return imgs

def dataArgument(data):
    """
        dataArguments
        data:
            imgs: N x 1 x W x H
            bbox: N x BBox
            landmarks: N x 10
    """
    pass

class BBox(object):
    '''
    人脸框
    '''
    def __init__(self, bbox):
        self.left = bbox[0]
        self.right = bbox[1]
        self.top = bbox[2]
        self.bottom = bbox[3]
        self.x = bbox[0]
        self.y = bbox[2]
        self.w = bbox[1] - bbox[0]
        self.h = bbox[3] - bbox[2]

    def expand(self, scale=0.05):
        bbox = [self.left, self.right, self.top, self.bottom]
        bbox[0] -= int(self.w * scale)
        bbox[1] += int(self.w * scale)
        bbox[2] -= int(self.h * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)

    def project(self, point):
        '''
        预处理
        :param point:
        :return:
        '''
        x = (point[0]-self.x) / self.w
        y = (point[1]-self.y) / self.h
        return np.asarray([x, y])

    def reproject(self, point):
        '''
        还原
        :param point:
        :return:
        '''
        x = self.x + self.w*point[0]
        y = self.y + self.h*point[1]
        return np.asarray([x, y])

    def reprojectLandmark(self, landmark):
        '''
        还原坐标点真实位置
        :param point:
        :return:
        '''
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p

    def projectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p

    def subBBox(self, leftR, rightR, topR, bottomR):
        leftDelta = self.w * leftR
        rightDelta = self.w * rightR
        topDelta = self.h * topR
        bottomDelta = self.h * bottomR
        left = self.left + leftDelta
        right = self.left + rightDelta
        top = self.top + topDelta
        bottom = self.top + bottomDelta
        return BBox([left, right, top, bottom])
