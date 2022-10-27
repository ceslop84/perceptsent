# PerceptSent

## Installation
```
git clone https://github.com/ceslop84/perceptsent
conda create --name py37_os python=3.7
conda activate py37_os
pip3 install -r requirements.txt
pip3 install --upgrade --no-cache-dir gdown
```
## How to Use

```
Run code samples: python3 main.py

Download the dataset: python 3 Dataset/download_images.py
```

## Abstract

Visual sentiment analysis is a challenging problem. Many datasets and approaches have been designed to foster breakthroughs in this trending research topic. However, most works scrutinize basic visual attributes from images, paying less attention to the viewers’ perceptions. 

Aiming to fill this gap, we present PerceptSent, a novel dataset for visual sentiment analysis that spans 5,000 images shared by users on social networks. Besides the sentiment opinion (positive, slightly positive, neutral, slightly negative, negative) expressed by every evaluator about each image analyzed, the dataset contains evaluator’s metadata (age, gender, socioeconomic status, education, and psychological hints) as well as subjective perceptions observed by the evaluator about the image — such as the presence of nature, violence, lack of maintenance, etc. 

Deep architectures and different problem formulations are explored using our dataset to combine visual and extra attributes for automatic sentiment analysis. We show evidence that evaluator’s perceptions, when correctly employed, are crucial in visual sentiment analysis, capable of improving accuracy and F-score performance in 30%, in average, reaching an impressive rate of
97.00% and 96.80%, respectively. Although at this point, we do not have automatic approaches to capture these perceptions, our results open up new investigation avenues. 

**Keywords**: visual sentiment analysis; subjective perception; deep networks; novel dataset.

## Supplementary material

To verify the supplementary material, please check the folder **"Results"**.

## Model Weights

To download the model weights for each execution, please download [here](rerere).
