from torch_geometric.explain import Explainer, ModelConfig
import torch
import torch.nn.functional as F
import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_embeddings
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data

import random
import numpy as np
from tqdm import tqdm

import os.path as osp
import os
from GNNs.GNN import GNN_Net
from Utils.pytorchtools import EarlyStopping
from F_explainer import FuzzyGNNExplainer
from utils import Draw_graph, get_meaned_inputs, Fix_edge_index, Remove_rows_by_index, extract_counterfactual_graph_embeddings
from Utils.FCM import Fcm_for_GNNX
from config import arg_parse
args = arg_parse()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = args.device


dataset_name = args.dataset
# load dataset
dataset = MoleculeNet('data', dataset_name)
torch.manual_seed(12345)
data_random = dataset.shuffle()
data_all = [data for data in data_random if data.num_nodes > 0]

batch_size = args.batch_size_E
epochs = args.epoch_E   #11


r_seed = random.seed(42)
train_dataset = data_all[:100]
test_dataset = data_all[:100]
fcm_data = random.sample(data_all, 500)

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
fcm_data_loader = DataLoader(fcm_data, batch_size=1, shuffle=True)

model_config = ModelConfig(mode='binary_classification', task_level='graph', return_type='probs')

# load GNN
gnn_model = GNN_Net(3)
model_name = args.dataset + '_gcn_model.pth'
model_save_path = osp.join('checkpoint/gnns', model_name)
gnn_model.load_state_dict(torch.load(model_save_path))
gnn_model.to(device)


#pre-learning the center of membership function
center_learner = Fcm_for_GNNX()


#get global mean graph embedding and edge embeddings
counterfactual_edge_embeddings = torch.FloatTensor([]).to(device)
sum_graph_embedding = None
for (i, G_i) in enumerate(fcm_data_loader):
    gnn_model.eval()
    G_i = G_i.to(device)
    z = gnn_model.get_node_reps(G_i.x, G_i.edge_index, G_i.edge_attr, G_i.batch)
    graph_embedding = global_mean_pool(z, G_i.batch)

    if sum_graph_embedding == None:
        sum_graph_embedding=graph_embedding
    else:
        sum_graph_embedding+=graph_embedding

    # Get counterfactual embedding for each edge

    counterfactual_edge_embeddings_i, _ = extract_counterfactual_graph_embeddings(G_i, gnn_model)
    counterfactual_edge_embeddings = torch.cat([counterfactual_edge_embeddings, counterfactual_edge_embeddings_i])


average_embedding = sum_graph_embedding / len(fcm_data_loader)

#get FCM centers and U to initialize membership function
U_fcm, centers_fcm, _= center_learner.Run_fcm(counterfactual_edge_embeddings, average_embedding)


std_devs = []
for i in range(centers_fcm.size(0)):
    # 用隶属度作为权重计算加权平均和加权标准差，使用隶属度作为权重，我们计算了每个聚类的加权平均和加权标准差。加权平均考虑了所有数据点对该聚类中心的隶属度，而加权标准差则反映了数据点在该聚类内的分布。
    weights = U_fcm[:, i]
    # 加权平均
    weighted_mean = torch.sum(counterfactual_edge_embeddings * weights.unsqueeze(1), dim=0) / torch.sum(weights)
    # 加权标准差
    diffs = counterfactual_edge_embeddings - weighted_mean.unsqueeze(0)
    weighted_sq_diffs = weights.unsqueeze(1) * (diffs ** 2)
    weighted_var = torch.sum(weighted_sq_diffs, dim=0) / torch.sum(weights)
    weighted_std_dev = torch.sqrt(weighted_var)

    std_devs.append(weighted_std_dev)

std_devs = torch.stack(std_devs)
#Explainer
FuzzyExplainer = FuzzyGNNExplainer(f_input_dim=4*gnn_model.get_GNN_output_dim(), centers=centers_fcm, sigma=std_devs)

explainer = Explainer(
    model=gnn_model,
    algorithm=FuzzyExplainer,
    explanation_type='phenomenon',
    edge_mask_type='object',
    model_config=model_config,
)

# earlystop
explainer_save_dir = 'checkpoint/Explainer'
explainer_name = args.dataset + '_explain_model.pth'
explainer_save_path = osp.join(explainer_save_dir, explainer_name)
early_stopping = EarlyStopping(10, verbose=True, path=explainer_save_path)



# Train against a variety of node-level or graph-level predictions:
print('Start training...')
for epoch in range(epochs):
    all_loss = 0.0
    bar = tqdm(train_data_loader)
    for data in bar:  # Indices to train against.
        bar.set_description(f'Epoch: {epoch}')
        data = data.to(device)
        loss = explainer.algorithm.train(gnn_model, data)
        all_loss += loss
        # print(loss)
        # print('Epoch: {0}  Train loss: {1}'.format(epoch + 1, loss))
    final_loss = all_loss / len(train_data_loader.dataset)
    print('Epoch: {0}  Train loss: {1:4f}'.format(epoch+1, final_loss))
    early_stopping(final_loss, FuzzyExplainer)
    if early_stopping.early_stop:
        print("Early stopping")
        # 结束模型训练
        break

# Testing:
FuzzyExplainer.reset_parameters()
FuzzyExplainer.load_state_dict(torch.load(explainer_save_path))
explainer = Explainer(
    model=gnn_model,
    algorithm=FuzzyExplainer,
    explanation_type='phenomenon',
    edge_mask_type='object',
    model_config=model_config,
)

all_explanations = []
all_f_plus = []
all_f_minus = []
all_f_plus_prob = []
all_f_minus_prob = []

import time

start_time = time.time()
for data in test_data_loader:
    data = data.to(device)
    f_plus, f_minus, f_plus_prob, f_minus_prob, explanation = explainer.algorithm.explaining(gnn_model, data)
    # print(data.y)
    all_f_plus.append(f_plus.tolist())
    all_f_minus.append(f_minus.tolist())
    all_f_plus_prob.append(f_plus_prob.tolist())
    all_f_minus_prob.append(f_minus_prob.tolist())
    all_explanations.append(explanation)
    # e_subgraph = explanation.get_complement_subgraph()
end_time = time.time()
run_time = end_time-start_time
single_run_time = run_time / len(test_data_loader)
print('Single_run_time：', single_run_time)

mean_f_plus = np.array(all_f_plus).mean(axis=0)
mean_f_minus = np.array(all_f_minus).mean(axis=0)
print('Fidelity +：', np.around(mean_f_plus, decimals=4))
print('Fidelity -：', np.around(mean_f_minus, decimals=4))

mean_f_plus_prob = np.array(all_f_plus_prob).mean(axis=0)
mean_f_minus_prob = np.array(all_f_minus_prob).mean(axis=0)
print('Fidelity_prob +：', np.around(mean_f_plus_prob, decimals=4))
print('Fidelity_prob -：', np.around(mean_f_minus_prob, decimals=4))


for i in all_explanations:
    Draw_graph(i)


