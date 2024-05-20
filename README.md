# LCPRE
This repository is built for our research "A Learning-Path-based Supervised Method for Concept Prerequisite Relations Extraction".

We will upload the other datasets soon.

In order to train LCPRE, you should modify the path of your pretrained language model in train.py and run the code below.
```
python train.py --datapath ./dataset/LB2  --learningpaths CC,CRC,CCRCC,CRCRC --max_intra 128 --batch_size 64 --epoch 50
```