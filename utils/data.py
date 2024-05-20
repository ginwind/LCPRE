import numpy as np
import torch
import random
from utils.rcGraph import readGraphMetaData
from random import choice
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MADataset(Dataset):
    def __init__(self, examples, labels, rcg, maxIntraPath, metaName, jumprandom=None):
        super(MADataset, self).__init__()
        self.examples = examples
        self.labels = labels
        intra_num = [maxIntraPath] * len(metaName)
        path_len = [len(each) for each in metaName]

        self.metapath1 = [torch.zeros(size=(len(examples), intra_num[i], path_len[i]), dtype=torch.int32) for i in range(0, len(metaName))]
        self.metapath2 = [torch.zeros(size=(len(examples), intra_num[i], path_len[i]), dtype=torch.int32) for i in range(0, len(metaName))]
        self.mask1 = [torch.zeros(size=(len(examples), intra_num[i]), dtype=torch.int32) for i in range(0, len(metaName))]
        self.mask2 = [torch.zeros(size=(len(examples), intra_num[i]), dtype=torch.int32) for i in range(0, len(metaName))]

        print("start traveling rc-graph.")
        for i in tqdm(range(0, len(metaName))):
            for j in range(0, len(examples)):
                start_paths, end_paths = rcg.nodeSpecificMetapath(examples[j][0], metaName[i], cache=False, jumprandom=jumprandom), rcg.nodeSpecificMetapath(examples[j][1], metaName[i], cache=False, jumprandom=jumprandom)
                if len(start_paths) != 0:
                    start_paths = torch.tensor(start_paths if len(start_paths) <= intra_num[i] else start_paths[: intra_num[i]], dtype=torch.int32)
                    self.metapath1[i][j][: start_paths.shape[0]] = start_paths
                    self.mask1[i][j][: start_paths.shape[0]] = 1
                if len(end_paths) != 0:
                    end_paths = torch.tensor(end_paths if len(end_paths) <= intra_num[i] else end_paths[: intra_num[i]], dtype=torch.int32)
                    self.metapath2[i][j][: end_paths.shape[0]] = end_paths
                    self.mask2[i][j][: end_paths.shape[0]] = 1

    def __getitem__(self, index):
        metapath1 = [each[index] for each in self.metapath1]
        metapath2 = [each[index] for each in self.metapath2] 
        mask1 = [each[index] for each in self.mask1]
        mask2 = [each[index] for each in self.mask2]
        labels = torch.tensor(self.labels[index], dtype=torch.float32)
        
        return metapath1, mask1, metapath2, mask2, labels
    
    def __len__(self):
        return len(self.examples)


def hashConceptPair(pairList):
    return str(pairList[0]) + " " + str(pairList[1])

def negExCreating(concepts, cc_edges, random_percent, neg_sample_num):
    """
    :param random_percent: percent of the examples which are choosed randomly, the rest(0.5 - random) are reverse examples.
    :param all_sample_num: the number of the total examples.

    :return np.array
    """ 
    # positive examples, negative examples.
    pos_examples, neg_examples = cc_edges, []

    preq_set = set(hashConceptPair(pairConcept) for pairConcept in pos_examples)

    while len(neg_examples) < neg_sample_num * random_percent:
        c1 = choice(concepts)
        c2 = choice(concepts)
        if c1 != c2 and hashConceptPair((c1, c2)) not in preq_set:
            neg_examples.append((c1, c2))

    while len(neg_examples) < neg_sample_num:
        line = choice(pos_examples)
        if hashConceptPair((line[1], line[0])) not in preq_set:
            neg_examples.append((line[1], line[0]))

    return np.array(pos_examples), np.array(neg_examples)

def shufflePosNeg(pos_examples, neg_examples):
    examples, labels = pos_examples+neg_examples, [1]*len(pos_examples)+[0]*len(neg_examples)
    c = list(zip(examples, labels))
    random.shuffle(c)
    examples, labels = zip(*c)
    examples = [tuple(each) for each in examples]
    return examples, labels