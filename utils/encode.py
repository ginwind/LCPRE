# TODO Long Text Embedding discussion.
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

def bertEncoding(path, bertPath, rcList, device):
    """
    encode nlp descriptions of rosource and concepts to features.
    :param path: save path of features.
    :param bertPath: the path of bert model.
    :param rcList: the list of descriptions of resources and concepts.
    :param device: torch.device.
    """
    print("descriptions loading finished.")
    tokenizer = BertTokenizer.from_pretrained(bertPath)
    model = BertModel.from_pretrained(bertPath).to(device).eval()
    print("model loading finished.")
    result = []
    with torch.no_grad():
        for each in tqdm(rcList):
            encoded_input = tokenizer(each, return_tensors='pt', 
                                    truncation=True, 
                                    padding=False, max_length=512,
                                    add_special_tokens=True).to(device)
            output = model(**encoded_input)
            result.append(output[0][0][0].unsqueeze(0).to(torch.device("cpu"))) # use [CLS] to represent the hole sentence.
            del output
            del encoded_input
    embedding = torch.cat(result, dim=0)
    print(embedding.shape)
    torch.save(embedding , path)
    return embedding