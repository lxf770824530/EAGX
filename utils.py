import torch
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import remove_isolated_nodes
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import remove_isolated_nodes, degree
from torch_geometric.nn import global_mean_pool

from networkx.algorithms.components import number_connected_components
from torch import Tensor, sigmoid
from typing import Optional, Union

import os


from config import arg_parse
args = arg_parse()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = args.device



def Del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)


def Remove_rows_by_index(tensor, indices_to_remove):
    #按索引从tensor中删除多行元素
    mask = torch.ones(tensor.size(0), dtype=torch.bool)
    mask[indices_to_remove] = False
    result = tensor[mask]
    return result


def Fix_edge_index(i,edge_index):
    rows_to_remove = [i]
    if len(edge_index) == 2:
        edge_index = torch.t(edge_index)


    u = edge_index[i][0].item()
    v = edge_index[i][1].item()
    # inverse_uv = torch.Tensor([v,u]).to(device)
    inverse_uv = torch.Tensor([v, u]).to(device)


    for j, edge in enumerate(edge_index.float()):
        if torch.equal(edge, inverse_uv):
            rows_to_remove.append(j)
            break
    new_edge_index = Remove_rows_by_index(edge_index, rows_to_remove)
    new_edge_index = torch.t(new_edge_index)
    return rows_to_remove, new_edge_index

def Get_edge_mask(deleted_edge, important_explanation_index, edge_index):
    edge_mask = [[] for k in range(0, len(edge_index[0]))]
    for i,ei in enumerate(important_explanation_index):
        if ei == True:
            for j in deleted_edge[i]:
                edge_mask[j]=True
        else:
            for j in deleted_edge[i]:
                edge_mask[j]=False
    return edge_mask

def Get_edge_degree(deleted_edge, degree, edge_index):
    edge_mask_list = [[] for k in range(0, len(edge_index[0]))]
    edge_mask_weight_dict = {}
    for i,de in enumerate(degree):
        for j in deleted_edge[i]:
            edge_mask_list[j]=de
    edge_index_T = torch.t(edge_index).tolist()
    for i, e in enumerate(edge_index_T):
        edge_mask_weight_dict[str(e)] = edge_mask_list[i]
    return edge_mask_list, edge_mask_weight_dict


def Draw_graph(Data,index=1):
    edge_index_T = torch.t(Data.edge_index).tolist()
    edge_mask_weight_dict = {}
    for i, e in enumerate(edge_index_T):
        edge_mask_weight_dict[str(e)] = Data.edge_mask[i].item()

    # for i, e in enumerate(edge_index_T):
    #     if edge_mask_weight_dict[str(e)] == 0.0:
    #         edge_mask_weight_dict[str([e[1], e[0]])] = 0.0

    G = to_networkx(Data, to_undirected=True, remove_self_loops=True)
    # print(edge_attr)
    pos = nx.spring_layout(G)
    for n in G.nodes:
        if Data.x[n].argmax().item() == 0:
            color = '#ffe600'   #黄色 O
            nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color=color)
    for (u, v, d) in G.edges(data=True):

        # print(u, v, edge_attr[i])
        # G.add_edge(u, v, weight=edge_attr[i])
        current_edge_weight = edge_mask_weight_dict[str([u,v])]
        if current_edge_weight>=0.5:
            edge_color = 'red'
            width = 2
        else:
            edge_color = 'blue'
            width = 1
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color= edge_color, width=width)
    # nx.draw(G)
    image_save_path = 'result/graph'+str(index)+'.png'
    plt.savefig(image_save_path)
    plt.show()
    plt.close('all')




def get_meaned_inputs(embedding: Tensor, edge_index: Tensor) -> Tensor:
    zs = (embedding[edge_index[0]] + embedding[edge_index[1]]) / 2
    return zs

def extract_counterfactual_graph_embeddings(graph, GNN):
    counterfactual_edge_embeddings_no_direct = torch.randn(graph.edge_index.size()[1], GNN.get_GNN_output_dim()).to(device)
    counterfactual_edge_embeddings = torch.FloatTensor([]).to(device)
    delete_edge_record = []
    deleted_edge = []
    for j in range(len(graph.edge_index[0])):
        if (j not in delete_edge_record) and len(delete_edge_record) < len(graph.edge_attr):
            deleted_index, edge_index = Fix_edge_index(j, graph.edge_index)
            if graph.edge_attr is not None:
                edge_attr = Remove_rows_by_index(graph.edge_attr, deleted_index)
            G_x = Data(x=graph.x, edge_index=edge_index, edge_attr=edge_attr)
            GNN.eval()
            z_c = GNN.get_node_reps(G_x.x, G_x.edge_index, G_x.edge_attr, G_x.batch)
            graph_embedding_c = global_mean_pool(z_c, G_x.batch)

            counterfactual_edge_embeddings = torch.cat([counterfactual_edge_embeddings, graph_embedding_c])
            counterfactual_edge_embeddings_no_direct[j] = graph_embedding_c
            counterfactual_edge_embeddings_no_direct[deleted_index[1]] = graph_embedding_c

            delete_edge_record.append(j)
            delete_edge_record.append(deleted_index[1])  # 一条边有两个索引。如：1,0  0,1
            deleted_edge.append(deleted_index)
    return counterfactual_edge_embeddings, counterfactual_edge_embeddings_no_direct


