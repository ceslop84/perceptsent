# PerceptSent

## Abstract

Visual sentiment analysis is a challenging problem. Many datasets and approaches have been designed to foster breakthroughs in this trending research topic. However, most works scrutinize only subsymbolic models through visual attributes of the evaluated images, paying less attention to the subjectivity of viewers' perceptions as a basis for neuro-symbolic systems. 

Aiming to fill this gap, we present PerceptSent, a novel dataset for visual sentiment analysis that spans 5,000 images shared by users on social networks. Besides the sentiment opinion (positive, slightly positive, neutral, slightly negative, negative) expressed by every evaluator about each image analyzed, the dataset contains evaluator's metadata (age, gender, socioeconomic status, education, and psychological hints) as well as perceptions observed by the evaluator about the image — such as the presence of nature, violence, lack of maintenance, etc. 

Deep architectures and different problem formulations are explored using our dataset to combine visual and extra attributes (external knowledge) for automatic sentiment analysis. We show evidence that evaluator's perceptionss, when correctly employed, are crucial in visual sentiment analysis, improving the F-score performance from 61% to an impressive rate above 97%. Although, at this point, we do not have automatic approaches to capture these perceptions, our results open up new investigation avenues.

**Keywords**: visual sentiment analysis; subjective perception; deep networks; novel dataset.

## Publication

C. Lopes, R. Minetto, M. Delgado and T. Silva, "PerceptSent - Exploring Subjectivity in a Novel Dataset for Visual Sentiment Analysis" in IEEE Transactions on Affective Computing, vol. , no. 01, pp. 1-15, 5555.
doi: 10.1109/TAFFC.2022.3225238

url: https://doi.ieeecomputersociety.org/10.1109/TAFFC.2022.3225238

## Installation
```
git clone https://github.com/ceslop84/perceptsent
conda create --name py37_os python=3.7
conda activate py37_os
pip3 install -r requirements.txt
```

## Download dataset

```
For more details regarding how to download the dataset, please verify the folder **"Dataset"**
```

## How to Use

```
python3 main.py

```

## Supplementary material

### Results datasheet

To verify the results datasheet, please check the folder **"Results"**.

### Model Weights

To download the model weights for each execution, please download [here](https://drive.google.com/file/d/1DGi2GePXJHb9XP2WRj6BzidCJCQCRBd4/view?usp=sharing).
