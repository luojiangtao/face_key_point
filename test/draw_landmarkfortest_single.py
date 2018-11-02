#coding:utf-8
'''
人脸关键点检测
'''
from os.path import join
import cv2
import sys
caffe_root = '/usr/bin/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class CNN(object):
    """
        生成CNN网络
    """

    def __init__(self, net, model):
        self.net = net
        self.model = model
        # 不存在则会报错
        self.cnn = caffe.Net(net, model, caffe.TEST)

    def forward(self, data, layer='fc2'):
        '''
        前向传播
        :param data: 数据
        :param layer: CNN网络层名称，默认全理解输出层
        :return:预测结果
        '''
        fake = np.zeros((len(data), 1, 1, 1))
        self.cnn.set_input_arrays(data.astype(np.float32), fake.astype(np.float32))
        self.cnn.forward()
        result = self.cnn.blobs[layer].data[0]
        # 2N --> Nx(2)
        t = lambda x: np.asarray([np.asarray([x[2*i], x[2*i+1]]) for i in range(len(x)//2)])
        result = t(result)
        return result


class BBox(object):
    """
        人脸框
    """
    def __init__(self, bbox):
        self.left = int(bbox[0])
        self.right = int(bbox[1])
        self.top = int(bbox[2])
        self.bottom = int(bbox[3])
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
        x = (point[0]-self.x) / self.w
        y = (point[1]-self.y) / self.h
        return np.asarray([x, y])

    def reproject(self, point):
        x = self.x + self.w*point[0]
        y = self.y + self.h*point[1]
        return np.asarray([x, y])

    def reprojectLandmark(self, landmark):
        '''
        坐标点还原为相对于图像的坐标
        :param landmark:
        :return:
        '''
        if not len(landmark) == 5:
            landmark = landmark[0]
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

    def cropImage(self, img):
        '''
        截取人脸框
        :param img:
        :return:
        '''
        return img[self.top:self.bottom+1, self.left:self.right+1]


class Landmarker(object):
    '''
    预测人脸关键点
    '''

    def __init__(self):
        '''
        初始化
        '''
        #model_path = join(PROJECT_ROOT, VERSION)
        deploy_path = "/python/face_key_point/prototxt"
        model_path = "/python/face_key_point/data_model"
        CNN_TYPES = ['LE1', 'RE1', 'N1', 'LM1', 'RM1', 'LE2', 'RE2', 'N2', 'LM2', 'RM2']
        level1 = [(join(deploy_path, '1_F_deploy.prototxt'), join(model_path, '1_F/_iter_100000.caffemodel'))]
        level2 = [(join(deploy_path, '2_%s_deploy.prototxt'%name), join(model_path, '2_%s/_iter_100000.caffemodel'%name)) \
                    for name in CNN_TYPES]
        self.level1 = [CNN(p, m) for p, m in level1]
        self.level2 = [CNN(p, m) for p, m in level2]

    def detectLandmark(self, image, bbox,two_level=True):
        '''
        在给定的人脸框内进行预测
        :param image: 图片
        :param bbox: 人类框
        :param two_level: 是否进行第二阶段的预测
        :return: 预测结果
        '''
        face = bbox.cropImage(image)
        face = cv2.resize(face, (39, 39))
        face = face.reshape((1, 1, 39, 39))
        face = self._processImage(face)
        # 第一阶段，对全脸做粗粒度提取
        landmark = self.level1[0].forward(face)
        if not two_level:
            return landmark

        # 第二阶段在第一阶段结果上取一小块区域微调
        landmark = self._level(image, bbox, landmark, self.level2, [0.16, 0.18])
        
        return landmark
    def _level(self, img, bbox, landmark, cnns, padding):
        '''
        第二阶段微调
        :param img: 图片
        :param bbox: 人脸框
        :param landmark:第一阶段预测的位置
        :param cnns: CNN网络
        :param padding: 内间距
        :return: 预测位置
        '''
        for i in range(5):
            x, y = landmark[i]
            patch, patch_bbox = self._getPatch(img, bbox, (x, y), padding[0])
            patch = cv2.resize(patch, (15, 15)).reshape((1, 1, 15, 15))
            patch = self._processImage(patch)
            d1 = cnns[i].forward(patch) # size = 1x2
            patch, patch_bbox = self._getPatch(img, bbox, (x, y), padding[1])
            patch = cv2.resize(patch, (15, 15)).reshape((1, 1, 15, 15))
            patch = self._processImage(patch)
            d2 = cnns[i+5].forward(patch)

            d1 = bbox.project(patch_bbox.reproject(d1[0]))
            d2 = bbox.project(patch_bbox.reproject(d2[0]))
            # 两种不同内间距预测结果求平均
            landmark[i] = (d1 + d2) / 2
        return landmark

    def _getPatch(self, img, bbox, point, padding):
        '''
        在给定的关键点附近取一小块区域
        :param img: 图片
        :param bbox: 人脸框
        :param point: 关键点
        :param padding: 内间距
        :return:
        '''
        
        point_x = bbox.x + point[0] * bbox.w
        point_y = bbox.y + point[1] * bbox.h
        patch_left = int(point_x - bbox.w * padding)
        patch_right = int(point_x + bbox.w * padding)
        patch_top = int(point_y - bbox.h * padding)
        patch_bottom = int(point_y + bbox.h * padding)
        patch = img[patch_top: patch_bottom+1, patch_left: patch_right+1]
        patch_bbox = BBox([patch_left, patch_right, patch_top, patch_bottom])
        return patch, patch_bbox
        

    def _processImage(self, imgs):
        """
            传给CNN前，先与处理，减均值除以方差
            imgs: N x 1 x W x H
        """
        imgs = imgs.astype(np.float32)
        for i, img in enumerate(imgs):
            m = img.mean()
            s = img.std()
            imgs[i] = (img - m) / s
        return imgs
    
def drawLandmark(img,  landmark):
    '''
    画出人脸关键点
    :param img:
    :param landmark:
    :return:
    '''
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 5, (0,255,0), -1)
    return img


def showResult(test_image,result_image,two_level=True):
    '''
    预测并画出人脸
    :param test_image: 测试图片位置
    :param result_image: 结果保存位置
    :param two_level: 是否进行第二阶段的预测
    :return:
    '''

    img = cv2.imread(test_image)
    # 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.resize(gray,(256,256))

    # 事先指定人脸框位置，小范围内能预测的更准确
    bbox = BBox([70 ,190 ,70,200])
    # 画人脸框
    cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)

    get_landmark = Landmarker()

    # 在给定的人脸框内进行预测
    final_landmark= get_landmark.detectLandmark(gray, bbox,two_level)

    # 坐标点还原为相对于图像的坐标
    final_landmark = bbox.reprojectLandmark(final_landmark)

    # 画出人脸关键点
    img = drawLandmark(img,  final_landmark)

    # 保存图像
    cv2.imwrite(result_image,img)

    img = Image.open(result_image)
    # 图像窗口名称
    plt.figure("Image")
    plt.imshow(img)
    #  关掉坐标轴为
    plt.axis('on')
    #  图像题目
    plt.title('result')
    # 显示图像
    plt.show()

if __name__ == '__main__':
    result_image = '/python/face_key_point/test/result/luojiangtao.jpg'
    test_image = '/python/face_key_point/test/test_image/luojiangtao.jpg'
    # 只进行第一阶段，全人脸预测
    showResult(test_image,result_image,two_level=False)
    # 在第一阶段结果上取一小块区域微调，获取更精确的位置
    showResult(test_image,result_image,two_level=True)