# SofaNet

Cross-center Early Sepsis Recognition by Medical Knowledge Guided Collaborative Learning for Data-scarce Hospitals

## Framework

![SofaNet] (img/SofaNet.png)


## Requirements

* Python 3.6
* torch 1.8
* cuda 1.11
* scikit-learn 0.24.1


## Quick Start

### Data Preparation

Because the two datasets we use require permission. Here we give the dataset links.

#### MIMIC-III

link: [https://physionet.org/content/mimiciii/1.4/](https://physionet.org/content/mimiciii/1.4/)

Sepsis labeling: [https://github.com/alistairewj/sepsis3-mimic](https://github.com/alistairewj/sepsis3-mimic)

#### Challenge

data link: [https://physionet.org/content/challenge-2019/1.0.0/](https://physionet.org/content/challenge-2019/1.0.0/)

We put a data example in `data` file.


### Run SofaNet

```
python train_SofaNet.py --data1 mimic --data2 challenge --alpha 0.5
```

If you want to train local model which concatnates with health status embeddings of SofaNet, you can train SofaNet_plus after get the SofaNet model.

```
python train_SofaNet_plus.py --data1 mimic --data2 challenge --alpha 0.5
```

**Attention**: the model will be saved in `save_dict` file. We have put the model we trained in the file.
