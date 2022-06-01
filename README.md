# PerceptSent

## Installation
```
git clone https://github.com/ceslop84/perceptsent
conda create --name py37_os python=3.7
conda activate py37_os
pip3 install -r requirements.txt
```
## How to Use

```
Usage: python3 main.py
```

**Parameters:**

**-m**: Name of model to train (case sensitive), or trained model. Currently the following models are supported:
- 'robust' - Implementation of CNN described by You _et al._ [[1]](https://arxiv.org/abs/1509.06041).
- 'vgg' - VGG16 [[2]](https://arxiv.org/abs/1409.1556).
- 'inception' - InceptionV3 [[3]](https://arxiv.org/abs/1512.00567).
- 'resnet' - ResNet50 [[4]](https://arxiv.org/abs/1512.03385).
- 'densenet' - DenseNet169 [[5]](https://arxiv.org/abs/1608.06993).
- 'xception' - Xception [[6]](https://arxiv.org/abs/1610.02357).

**-a**: Directory containing attributes files in txt format. Must be defined when loading a model trained with attributes. If omitted when training a model, attributes will not be used.

**-n**: If using attributes this must be set to attributes length. Default value is 102, length of SUN attributes.

**-t**: File with list of images and labels to use on train (see details below).

**-e**: File with list of images and labels to evaluate (see details below).

**-c**: File with list of images to classify (see details below) or directory with images to classify.

**-k**: Number of folds to use for k-fold cross validation on train. Default value is 5.


The image list passed to '-t', '-e' and '-c' arguments is a text file where each line has the path to an image and correspondent label, for classification labels are optional.
Example of image list:
```
Images/1495676.jpg 1
Images/36282362.jpg 2
Images/994471.jpg 0
```

Labels are:
- 0 for Negative
- 1 for Neutral
- 2 for Positive

### Extract SUN Attributes

```
python3 places365.py Images
```

*Images* can be a file that list the images from where the attributes will be extracted, or a directory containing the images. This requires python version 3.6.x.

### Examples

#### Train Model
Train *robust* model using *SUN* attributes using all images in *labels.txt*:
```
python3 OutdoorSent.py -m robust -a SUN -t labels.txt -k 1
```

Train *VGG* model without attributes using 5-fold cross-validation, note that *-k 5* argument is optional.
```
python3 OutdoorSent.py -m vgg -t labels.txt -k 5
```

#### Evaluate Images
Evaluate classification of *inception* model using *YOLO* attributes.
```
python3 OutdoorSent.py -m Weights/inception_T_0.h5 -a YOLO -e labels.txt -n 9418
```

#### Classify Images
Classify files in '*Images*' directory using *resnet* model without attributes.
```
python3 OutdoorSent.py -m Weights/resnet_F_.h5 -c Images
```

### References

[1] You, Q., Luo, J., Jin, H., & Yang, J. (2015, February). Robust image sentiment analysis using progressively trained and domain transferred deep networks. In Twenty-ninth AAAI conference on artificial intelligence.

[2] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

[3] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2818-2826).

[4] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

[5] Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4700-4708).

[6] Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1251-1258).
