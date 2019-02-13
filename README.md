# 人脸关键点检测
基于caffe 开发的人脸关键点检测
- 只进行第一阶段，全人脸预测的结果

![效果截图](https://raw.githubusercontent.com/luojiangtao/face_key_point/master/1.png)  

- 在第一阶段结果上取一小块区域微调，获取更精确的位置

![效果截图](https://raw.githubusercontent.com/luojiangtao/face_key_point/master/2.png)  
## 我使用的版本
*   python 3.6
*   ubuntu 18.04.1
*   caffe 1.0.0