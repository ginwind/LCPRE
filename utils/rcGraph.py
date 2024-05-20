import os
import shutil
import pickle
from tqdm import tqdm
import random
import pandas as pd
import networkx as nx
from collections import defaultdict

def readGraphMetaData(path):
    """
    Get the graph description data with droping duplicates. We replace the resources and courses except descriptions with index. The elements in cc, rc, rr lists are Python Tuple.

    :param path: parent data path.
    return: resources, concepts, cc, rc, rr, descriptions of rc(all types of above data are Python List)
    """
    rc_edges = pd.read_csv(path + "/rc.csv").drop_duplicates()
    cc_edges = pd.read_csv(path + "/cc.csv").drop_duplicates()
    rr_edges = pd.read_csv(path + "/rr.csv").drop_duplicates()

    if len(rr_edges) != 0:
        # TODO resources should duplicate some characters.
        resources = pd.concat([rc_edges["Courses"], rr_edges["0"], rr_edges["1"]]).drop_duplicates().to_list()
    else:
        resources = rc_edges["Courses"].drop_duplicates().to_list()
    concepts = pd.concat([rc_edges["Concepts"], cc_edges["Concept1"], cc_edges["Concept2"]]).drop_duplicates().to_list()
    
    resourceDes = pd.read_csv(path + "/courses.csv")
    des = [str(resourceDes[resourceDes["Courses"]==each]["Descriptions"].values) for each in resources]

    idx2nodes = resources + concepts
    resource_text = des
    concept_text = concepts
    node2Idx = {}
    for each in idx2nodes:
        node2Idx[each] = len(node2Idx)

    resources = [node2Idx[each] for each in resources]
    concepts = [node2Idx[each] for each in concepts]
    cc_edges = [(node2Idx[each[1]], node2Idx[each[0]]) for each in cc_edges.values.tolist()]
    rc_edges = [(node2Idx[each[0]], node2Idx[each[1]]) for each in rc_edges.values.tolist()]
    rr_edges = [(node2Idx[each[0]], node2Idx[each[1]]) for each in rr_edges.values.tolist()]

    return resources, concepts, cc_edges, rc_edges, rr_edges, des, node2Idx, resource_text, concept_text, idx2nodes

class RCGraph:
    def __init__(self, rc_nx_graph: nx.DiGraph, max_intra: int, data_path: str):
        self.rc_nx_graph = rc_nx_graph
        self.max_intra = max_intra
        self.data_path = data_path
        self.schema2paths = {}
        node2degree = dict(rc_nx_graph.degree())
        self.concepts = [node for node in list(self.rc_nx_graph.nodes) if self.rc_nx_graph.nodes[node]["type"] == "C"]
        self.resources = [node for node in list(self.rc_nx_graph.nodes) if self.rc_nx_graph.nodes[node]["type"] == "R"]
        self.conceptWeights = [1 / (node2degree[node] + 1) for node in self.concepts]
        self.resourceWeights = [1 / (node2degree[node] + 1) for node in self.resources]

    
    def clearCache(self):
        shutil.rmtree(f"{self.data_path}/cache")
        os.mkdir(f"{self.data_path}/cache")

    def saveAsDic(self, schema):
        node2path = defaultdict(list)
        concepts = [node for node in list(self.rc_nx_graph.nodes) if self.rc_nx_graph.nodes[node]["type"] == "C"]
        for each in concepts:
            paths = []
            self.travelAllPaths(1, schema, [each], paths)
            node2path[each] = paths
        self.schema2paths[schema] = node2path
        with open(f"{self.data_path}/cache/{schema}.pkl", "wb") as f:
            pickle.dump(node2path, file=f)
        return node2path
    
    def loadDic(self, schema):
        with open(f"{self.data_path}/cache/{schema}.pkl", "rb") as f:
            self.schema2paths[schema] = pickle.load(file=f)

    def travelAllPaths(self, i, schema, path, paths):
        if i == len(schema):
            if path[-1] != path[0]:
                paths.append(path)
            return
        neighbors = [node for node in list(self.rc_nx_graph.neighbors(path[i - 1])) if self.rc_nx_graph.nodes[node]["type"] == schema[i]]
        for each in neighbors:
            self.travelAllPaths(i + 1, schema, path + [each], paths)
        
    def maxIntraTra(self, i, schema, path, paths, jumprandom):
        if len(paths) > self.max_intra:
            return
        if i == len(schema):
            if path[-1] != path[0]:
                paths.append(path)
            return
        if jumprandom == None:
            neighbors = [node for node in list(self.rc_nx_graph.neighbors(path[i - 1])) if self.rc_nx_graph.nodes[node]["type"] == schema[i]]
            random.shuffle(neighbors)
            for each in neighbors:
                self.maxIntraTra(i + 1, schema, path + [each], paths, jumprandom)
            return
        neighbors = [node for node in list(self.rc_nx_graph.neighbors(path[i - 1])) if self.rc_nx_graph.nodes[node]["type"] == schema[i]]
        jumpConfig = (self.concepts, self.conceptWeights) if schema[i] == "C" else (self.resources, self.resourceWeights)
        if len(neighbors) != 0:
            a = random.random()
            if a > jumprandom:
                self.maxIntraTra(i + 1, schema, path + [random.choice(neighbors)], paths, jumprandom)
            else:
                self.maxIntraTra(i + 1, schema, path + [random.choices(jumpConfig[0], weights=jumpConfig[1], k=1)[0]], paths, jumprandom)
        else:
            self.maxIntraTra(i + 1, schema, path + [random.choices(jumpConfig[0], weights=jumpConfig[1], k=1)[0]], paths, jumprandom)

    def nodeSpecificMetapath(self, node, schema, cache=False, jumprandom=0.2):
        if cache:
            if os.path.exists(f"{self.data_path}/cache/{schema}.pkl"):
                if schema not in self.schema2paths.keys():
                    self.loadDic(schema)
            else:
                self.saveAsDic(schema)
            return random.sample(self.schema2paths[schema][node], k=min(self.max_intra, len(self.schema2paths[schema][node])))
        else:
            paths = []
            self.maxIntraTra(1, schema, [node], paths, jumprandom)
            return paths


def createRCGraph(resources, concepts, cc_edges, rc_edges, rr_edges, max_intra, data_path):
    """
    create a heterogeneous graph using path after preprocessing in order to get path.
    :param path:
    :return:
    """
    rc_nx_graph = nx.DiGraph()
    # Build Resources Nodes and RR edges.
    for resource in resources:
        rc_nx_graph.add_node(resource, type="R")
    for each in rr_edges:
        rc_nx_graph.add_edge(each[0], each[1], type="RR")

    # Build Concepts Nodes and CC edges.
    for concept in concepts:
        rc_nx_graph.add_node(concept, type="C")
    for each in cc_edges:
        rc_nx_graph.add_edge(each[0], each[1], type="CC")

    # Build RC edges.
    for i in range(0, len(rc_edges)):
        each = rc_edges[i]
        rc_nx_graph.add_edge(each[0], each[1], type="CR")
        rc_nx_graph.add_edge(each[1], each[0], type="RC")

    rc_graph = RCGraph(rc_nx_graph=rc_nx_graph, max_intra=max_intra, data_path=data_path)
    print(f"finish building rc graph which contains {len(resources)} resources and {len(concepts)} concepts.")
    print(f"RR edges:{len(rr_edges)}, RC edges:{len(rc_edges)}, CC edges:{len(cc_edges)}.")
    return rc_graph