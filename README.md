[cars-yolo-output]: examples/assets/cars.gif "Sample Output with YOLO"
[cows-tf-ssd-output]: examples/assets/cows.gif "Sample Output with SSD"

# Multi-object trackers in Python
Easy to use implementation of various multi-object tracking algorithms.

[![DOI](https://zenodo.org/badge/148338463.svg)](https://zenodo.org/badge/latestdoi/148338463)


`YOLOv3 + CentroidTracker` |  `TF-MobileNetSSD + CentroidTracker`
:-------------------------:|:-------------------------:
![Cars with YOLO][cars-yolo-output]  |  ![Cows with tf-SSD][cows-tf-ssd-output]
Video source: [link](https://flic.kr/p/L6qyxj) | Video source: [link](https://flic.kr/p/26WeEWy)


## Available Multi Object Trackers

```
CentroidTracker
IOUTracker
CentroidKF_Tracker
SORT
```

## Available OpenCV-based object detectors:

```
detector.TF_SSDMobileNetV2
detector.Caffe_SSDMobileNet
detector.YOLOv3
```

## Installation

Pip install for OpenCV (version 3.4.3 or later) is available [here](https://pypi.org/project/opencv-python/) and can be done with the following command:

```
git clone https://github.com/adipandas/multi-object-tracker
cd multi-object-tracker
pip install -r requirements.txt
pip install -e .
```

**Note - for using neural network models with GPU**  
For using the opencv `dnn`-based object detection modules provided in this repository with GPU, you may have to compile a CUDA enabled version of OpenCV from source.  
* To build opencv from source, refer the following links:
[[link-1](https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html)],
[[link-2](https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/)]

## How to use?: Examples

Please refer [examples](https://github.com/adipandas/multi-object-tracker/tree/master/examples) folder of this repository.
You can clone and run the examples as shown [here](examples/readme.md).

## Pretrained object detection models

You will have to download the pretrained weights for the neural-network models. 
The shell scripts for downloading these are provided [here](https://github.com/adipandas/multi-object-tracker/tree/master/examples/pretrained_models) below respective folders.
Please refer [DOWNLOAD_WEIGHTS.md](DOWNLOAD_WEIGHTS.md) for more details.

### Notes
* There are some variations in implementations as compared to what appeared in papers of `SORT` and `IoU Tracker`.
* In case you find any bugs in the algorithm, I will be happy to accept your pull request or you can create an issue to point it out.

## References and Credits

Please see [REFERENCES.md](REFERENCES.md).

## Citation

If you use this repository in your work, please consider citing it with:
```
@misc{multiobjtracker_amd2018,
  author = {Deshpande, Aditya M.},
  title = {Multi-object trackers in Python},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/adipandas/multi-object-tracker}},
}
```

```
@software{aditya_m_deshpande_2020_3951169,
  author       = {Aditya M. Deshpande},
  title        = {Multi-object trackers in Python},
  month        = jul,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.3951169},
  url          = {https://doi.org/10.5281/zenodo.3951169}
}
```
