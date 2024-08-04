# LCPRE
This repository includes the code and demo of our research ["A Learning-path based Supervised Method for Concept Prerequisite Relations Extraction in Educational Data"](https://doi.org/10.1145/3627673.3679597) (the hyperlink will become active after publication in the ACM Digital Library), which is accepted by CIKM 2024.

## Environment Requirements
```
numpy>=1.25.1
torch>=1.13.1
pandas>=2.0.3
tqdm==4.65.0
sklearn>=1.3.0
transformers>=4.30.2
networkx>=3.0
```


## Dataset
The benchmark dataset used in our paper is located at ./dataset. If you have another prerequisite concept dataset, you can create a new folder under ./dataset, which includes the files below.
```
Name_of_Your_New_Dataset
??? cc.csv
??? courses.csv
??? rr.csv
```
The cc.csv includes your concept prerequisite relations.    
The courses.csv includes the textual descriptions of each course.   
If you have prerequisite relations between courses, you can put it in rr.csv. Otherwise, you need to create an empty file named rr.csv.  

We use TFIDF to connect the concept and course. To achieve that purpose, we modify the datapath and the threhold of TFIDF in tfidf.py and run it.
```
# modify the datapath and the threhold of TFIDF here.
path = "./dataset/UCD"
threshold = 0.05
```

Besides, we use a complex negative examples sampling approach, which enhances the robustness of our model, as detailed in the Section 4.2 of our paper. The dataset.csv includes both the positive and negative examples used to training and evaluating.
 
## Training and Evaluating
In order to train and evaluate LCPRE, you should modify the path of your pretrained language model in train.py.
```
# modify your pretrained language model path here
bert_path = None
```

Then run the code below.
```
python train.py --datapath ./dataset/LB2  --learningpaths CC,CRC,CCRCC,CRCRC --max_intra 128 --batch_size 64 --epoch 50

python train.py --datapath ./dataset/UCD --metapaths CC,CRC,CRRC --max_intra 256 --batch_size 32 --epoch 50 

python train.py --datapath ./dataset/MOOC  --metapaths CC,CRC,CRRC,CCRCC,CRCRC --max_intra 128 --batch_size 64 --epoch 50 
```