# 应用领域

本文立足于**将超声悬浮技术应用于超疏水表面上的液滴操控系统**，并在此基础上搭建**以机器视觉**为辅助的三轴式液滴操控系统。本文目的是利用神经网络实现液滴的目标检测与目标跟踪问题，实现液滴的自动化操控，并且提高液滴的操控精度。通过一个轻量化网络，使得在边缘计算设备上也能运行精准的液滴目标检测与跟踪算法。

## Available Object Detector

```
NanoDet-Plus
```

## Available Multi Object Trackers

```
CentroidTracker
IOUTracker
CentroidKF_Tracker
SORT
```

## Installation

```
git clone https://github.com/vvEverett/multi-object-tracker.git
cd multi-object-tracker
pip install -r requirements.txt
# pip install -e .
python setup.py develop
python setup_nanodet.py develop
```

## How to use?

运行main.py即可开启对test.avi的液滴目标检测与跟踪。

