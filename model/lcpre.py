import math
import torch
import torch.nn.functional as F
from torch import nn

class NodeTransformation(nn.Module):
    def __init__(self, embedding_matrix, num_resource, in_dim, out_dim, attn_drop=0.5):
        """
        load embedding matrix from embedding_path, we assume that the resource index is smaller than the concept index.
        """
        super(NodeTransformation, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.embedding.weight.requires_grad = False
        self.attn_drop = nn.Dropout(attn_drop)
        self.num_resource = num_resource
        
        self.resourceTrans = nn.Linear(in_dim, out_dim)
        self.conceptTrans = nn.Linear(in_dim, out_dim)
    
    def forward(self, index):
        # index: batch_size * path_len
        x = self.attn_drop(self.embedding(index)) # batch_size * path_len * in_dim
        x = torch.where(index.unsqueeze(-1) < self.num_resource, self.resourceTrans(x), self.conceptTrans(x)) # batch_size * path_len * out_dim
        return x

class LCPRE(nn.Module):
    def __init__(self, embedding_matrix, metaname, num_resource,
                 embedding_dim=768, projection_dim=128, num_heads=4, node_dim=128,
                 attn_vec_dim=128, out_dim=128, attn_drop=0.5, agg_type="att"):
        """
        :param num_metapath: the number of metapaths when considering a pair of nodes.
        :param embedding_path: the matrix of embedding features.
        :param num_resource: the number of resources.
        """
        super(LCPRE, self).__init__()
        self.embedding = NodeTransformation(embedding_matrix, num_resource, embedding_dim, projection_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        num_metapath = len(metaname)
        max_path_len = max(len(each) for each in metaname)

        self.startIntra = nn.ModuleList()
        self.endIntra = nn.ModuleList()
        for i in range(0, num_metapath):
            self.startIntra.append(IntraMA(agg_type, num_heads=num_heads, node_dim=node_dim, max_path_len=max_path_len))
            self.endIntra.append(IntraMA(agg_type, num_heads=num_heads, node_dim=node_dim, max_path_len=max_path_len))
        self.startInter = InterMA(node_dim, attn_vec_dim, num_metapath)
        self.endInter = InterMA(node_dim,attn_vec_dim, num_metapath)
        self.fc_concept1 = nn.Linear(node_dim, out_dim, bias=True)
        self.fc_concept2 = nn.Linear(node_dim, out_dim, bias=True)

        # weight initialization
        nn.init.xavier_normal_(self.fc_concept1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_concept2.weight, gain=1.414)

    def forward(self, metapath1, mask1, metapath2, mask2):
        """
        :param metapaths: list[batch_size * n * path_len]
        :param masks: list[batch_size * n]
        """
        start_outs = []
        for i, data in enumerate(zip(metapath1, mask1, self.startIntra)):
            # step 1. embedding and transformation.
            metapath, mask, intraMP = data
            instances = self.embedding(metapath)
            mask = mask.unsqueeze(-1)

            # step 2. intra-metapath aggresive.
            start_outs.append(intraMP(instances, mask))

        end_outs = []
        for i, data in enumerate(zip(metapath2, mask2, self.endIntra)):
            # step 1. embedding and transformation.
            metapath, mask, intraMP = data
            instances = self.embedding(metapath)
            mask = mask.unsqueeze(-1)

            # step 2. intra-metapath aggresive.
            end_outs.append(intraMP(instances, mask))
        # step 3. inter-metapath aggresive.
        h1 = self.fc_concept1(self.startInter(start_outs))
        h2 = self.fc_concept2(self.endInter(end_outs))

        return h1, h2
    
    def play(self, metapath1, mask1, metapath2, mask2, adjnodes, dm_index):
        """
        :param metapaths: list[batch_size * n * path_len]
        :param masks: list[batch_size * n]
        """
        start_outs = []
        index = 0
        for metapath, mask, intraMP in zip(metapath1, mask1, self.startIntra):
            # step 1. embedding and transformation.
            instances = self.attn_drop(self.embedding(metapath))
            mask = mask.unsqueeze(-1)

            # step 2. intra-metapath aggresive.
            start_outs.append(intraMP.play(instances, mask, metapath, adjnodes, index, dm_index))
            index += 1

        end_outs = []
        index = 0
        for metapath, mask, intraMP in zip(metapath2, mask2, self.endIntra):
            # step 1. embedding and transformation.
            instances = self.attn_drop(self.embedding(metapath))
            mask = mask.unsqueeze(-1)

            # step 2. intra-metapath aggresive.
            end_outs.append(intraMP(instances, mask))
            index += 1
        
        # step 3. inter-metapath aggresive.
        h1 = self.fc_concept1(self.startInter.play(start_outs, adjnodes))
        h2 = self.fc_concept2(self.endInter(end_outs))

        return h1, h2

class InterMA(nn.Module):
    def __init__(self, node_dim, attn_vec_dim, num_metapath):
        super(InterMA, self).__init__()
        self.fc1 = nn.ModuleList()
        for i in range(0, num_metapath):
            self.fc1.append(nn.Linear(node_dim, attn_vec_dim, bias=True))
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)

        [nn.init.xavier_normal_(each.weight, gain=1.414) for each in self.fc1]
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)
    
    def forward(self, metapath_outs):
        beta = []
        for i, metapath_out in enumerate(metapath_outs):
            fc1 = F.relu(self.fc1[i](metapath_out))
            fc1_mean = torch.mean(fc1, dim=0)
            fc2 = self.fc2(fc1_mean)
            beta.append(fc2)
        beta = torch.cat(beta, dim=0)
        beta = F.softmax(beta, dim=0)
        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)
        metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in metapath_outs]
        metapath_outs = torch.cat(metapath_outs, dim=0)
        h = torch.sum(beta * metapath_outs, dim=0)
        return h
    
    def play(self, metapath_outs, adj_nodes):
        beta = []
        for i, metapath_out in enumerate(metapath_outs):
            fc1 = F.relu(self.fc1[i](metapath_out))
            fc1_mean = torch.mean(fc1, dim=0)
            fc2 = self.fc2(fc1_mean)
            beta.append(fc2)
        beta = torch.cat(beta, dim=0)
        beta = F.softmax(beta, dim=0)
        playbeta = beta.flatten().cpu().detach().numpy().tolist()

        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)
        metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in metapath_outs]
        metapath_outs = torch.cat(metapath_outs, dim=0)
        h = torch.sum(beta * metapath_outs, dim=0)
        for key, value in adj_nodes.items():
            value = [value[i] * playbeta[i] for i in range(0, len(value))]
            adj_nodes[key] = value
        return h
    

class IntraMA(nn.Module):
    def __init__(self, rnn_type, 
                 num_heads=4, node_dim=128, max_path_len=7):
        super(IntraMA, self).__init__()
        self.rnn_type = rnn_type
        self.num_heads = num_heads
        self.node_dim = node_dim
        self.multiheadAtt = AttentionWithPosEncoding(max_path_len, num_heads, node_dim)

        self.attn1 = nn.Linear(node_dim, num_heads, bias=False)
        self.attn2 = nn.Parameter(torch.empty(size=(1, num_heads, node_dim // num_heads)))

        nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
        nn.init.xavier_normal_(self.attn2.data, gain=1.414)

    def forward(self, instances, masks):
        """
        :param key & query & value: batch_size * n * path_len * node_dim
        :param masks: batch_size * n * 1   [1 is true 0 is false]
        """
        batch_size, N, path_len, _ = instances.shape
        instances = instances.view(batch_size * N, path_len, -1)
        att_outs = self.multiheadAtt(instances) # (batch_size * N) * path_len * node_dim

        start = att_outs[:, 0, :]
        mpNeighbors = att_outs[:, -1, :].view(-1, self.num_heads, self.node_dim // self.num_heads) # (batch_size * N) * num_heads * (node_dim // num_heads)

        # step 1. neighbor nodes aggresive leaky_relu.
        center_features = start.view(batch_size * N, -1) # (batch_size * N) * node_dim
        a1 = self.attn1(center_features)  # E x num_heads
        a2 = (mpNeighbors * self.attn2).sum(dim=-1)  # E x num_heads
        a = F.leaky_relu((a1 + a2)).view(batch_size, N, -1)  # batch_size * N * num_heads

        # step 2. masked attention score.
        score = F.softmax(torch.where(masks == 1, a, 0), dim=1).unsqueeze(dim=-1) # batch_size * N * num_heads * 1
        weighted_sum = F.relu(torch.sum((mpNeighbors.view(batch_size, N, self.num_heads, -1) * score), dim=1)) # batch_size * num_heads * node_dim
        return weighted_sum.view(batch_size, -1)
    
    def play(self, instances, masks, metapath, adjnodes, index, dm_index):
        """
        :param key & query & value: batch_size * n * path_len * node_dim
        :param masks: batch_size * n * 1   [1 is true 0 is false]
        """
        batch_size, N, path_len, _ = instances.shape
        instances = instances.view(batch_size * N, path_len, -1)
        att_outs = self.multiheadAtt(instances) # (batch_size * N) * path_len * node_dim

        start = att_outs[:, 0, :]
        mpNeighbors = att_outs[:, -1, :].view(-1, self.num_heads, self.node_dim // self.num_heads) # (batch_size * N) * num_heads * (node_dim // num_heads)

        # step 1. neighbor nodes aggresive leaky_relu.
        center_features = start.view(batch_size * N, -1) # (batch_size * N) * node_dim
        a1 = self.attn1(center_features)  # E x num_heads
        a2 = (mpNeighbors * self.attn2).sum(dim=-1)  # E x num_heads
        a = F.leaky_relu((a1 + a2)).view(batch_size, N, -1)  # batch_size * N * num_heads

        # step 2. masked attention score.
        score = F.softmax(torch.where(masks == 1, a, 0), dim=1).unsqueeze(dim=-1) # batch_size * N * num_heads * 1
        weighted_sum = F.relu(torch.sum((mpNeighbors.view(batch_size, N, self.num_heads, -1) * score), dim=1)) # batch_size * num_heads * node_dim
        
        play_score = torch.mean(score.view(N, self.num_heads), dim=1)
        for i in range(0, metapath.shape[1]):
            if metapath[0][i][0].item() == dm_index:
                adjnodes[metapath[0][i][-1].item()][index] += play_score[i].item()
        
        return weighted_sum.view(batch_size, -1)



class AttentionWithPosEncoding(nn.Module):
    def __init__(self, max_path_len, num_heads=4, node_dim=128):
        super(AttentionWithPosEncoding, self).__init__()
        self.embedding = nn.Embedding(max_path_len, node_dim)
        self.multiheadAtt = nn.MultiheadAttention(node_dim, num_heads, batch_first=True)

        self.num_heads = num_heads
        self.register_buffer("position_ids", torch.arange(max_path_len))
    
    def forward(self, instances):
        path_len = instances.shape[1]
        position_ids = self.position_ids[: path_len]
        instances = instances + self.embedding(position_ids)
        
        value_out, _ = self.multiheadAtt(instances, instances, instances)
        return value_out
  





    
