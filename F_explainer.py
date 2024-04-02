import torch
import logging
from torch import Tensor, sigmoid
from typing import Optional, Union
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU, Sequential
from torch_geometric.nn import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explanation
from torch_geometric.data import Data
from torch_geometric.utils import get_embeddings
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import (
    ExplanationType,
    ModelMode,
    ModelTaskLevel,
)
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from utils import extract_counterfactual_graph_embeddings

import os.path as osp
import os

from Utils.Attributor import Attributor
from config import arg_parse
args = arg_parse()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = args.device



class FuzzyGNNExplainer(ExplainerAlgorithm):
    coeffs = {
        'edge_size': 0.05,
        'edge_ent': 0.01,
        'bias': 0.0,
        'Regularation_loss_weight': 0.3
    }

    def __init__(self, f_input_dim, centers, sigma, **kwargs):
        super().__init__()
        self.coeffs.update(kwargs)

        # self.f_output_dim = 64
        self.f_input_dim = f_input_dim

        self.membership_fun = Attributor(self.f_input_dim, centers, sigma).to(device)

        # self.mlp_fuzzy = Sequential(
        #     FuzzyLayer(self.f_input_dim, self.f_output_dim, centers, sigma),
        #     ReLU(),
        #     Linear(self.f_output_dim, 2),
        #     ReLU(),
        #     Linear(2, 1),
        # ).to(device)

        self.optimizer = torch.optim.Adam(self.membership_fun.parameters(), lr=0.001)
        # self.scheduler = StepLR(optimizer=self.optimizer, step_size=10, gamma=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer,
                                      mode='min',
                                      factor=0.8,
                                      patience=5,
                                      min_lr=1e-4
                                      )



    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.membership_fun)


    def train(
        self,
        model: torch.nn.Module,
        graph,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):


        if isinstance(graph.x, dict) or isinstance(graph.edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided "
                                 f"in '{self.__class__.__name__}' for "
                                 f"node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' "
                                 f"argument in '{self.__class__.__name__}'")
        self.optimizer.zero_grad()

        # z = get_embeddings(model, graph.x, graph.edge_index, graph.edge_attr, graph.batch)[-1]
        z = model.get_node_reps(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        graph_embedding = global_mean_pool(z, graph.batch)
        edge_batch = graph.batch[graph.edge_index[0, :]]

        cat_input = self._get_inputs(z, graph.edge_index, index)

        cause_edge_mask = torch.FloatTensor([]).to(device)
        for i in range((graph.batch[-1]+1).item()):
            node_indicator = (graph.batch == i).bool()
            edge_indicator = (edge_batch == i).bool()
            edge_embs = cat_input[edge_indicator]

            G_i_x = graph.x[node_indicator]
            G_i_edge_index = graph.edge_index[:,edge_indicator] - graph.edge_index[:,edge_indicator].min()
            G_i_edge_attr = graph.edge_attr[edge_indicator]
            G_i = Data(x=G_i_x, edge_index=G_i_edge_index, edge_attr=G_i_edge_attr)
            _, G_i_counters = extract_counterfactual_graph_embeddings(G_i, model)

            logits = self.membership_fun(edge_embs, G_i_counters).view(-1)
            cause_i_edge_mask = self._concrete_sample(logits)
            cause_edge_mask = torch.cat([cause_edge_mask, cause_i_edge_mask])

        # cause part
        set_masks(model, cause_edge_mask, graph.edge_index, apply_sigmoid=True)
        # cause_node_embeddings = get_embeddings(model, graph.x, graph.edge_index, graph.edge_attr, graph.batch)[-1]
        cause_node_embeddings = model.get_node_reps(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        cause_graph_embedding = global_mean_pool(cause_node_embeddings, graph.batch)
        cause_graph_y_hat = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        clear_masks(model)

        #context part
        context_edge_mask = 1-cause_edge_mask
        set_masks(model, context_edge_mask, graph.edge_index, apply_sigmoid=True)
        # context_node_embeddings = get_embeddings(model, graph.x, graph.edge_index, graph.edge_attr, graph.batch)[-1]
        context_node_embeddings = model.get_node_reps(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        context_graph_embedding = global_mean_pool(context_node_embeddings, graph.batch)
        context_graph_y_hat = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        clear_masks(model)

        y = F.one_hot(graph.y.long(), num_classes=2).squeeze(1)

        loss = self._loss(cause_graph_y_hat, context_graph_y_hat, y, cause_edge_mask)

        # print([aa.grad for aa in self.optimizer.param_groups[0]['params']])
        # print([aa.grad for aa in self.optimizer.param_groups[1]['params']])
        # for name, parameters in self.mlp_fuzzy.named_parameters():
        #     print(name, ':', parameters, parameters.size())
        loss.backward(retain_graph=True)
        # for name, parameters in self.mlp_fuzzy.named_parameters():
        #     print(name, ':', parameters, parameters.size())
        self.optimizer.step()
        self.scheduler.step(loss)
        # lr = self.scheduler.optimizer.param_groups[0]['lr']
        # for name, parameters in self.mlp_fuzzy.named_parameters():
        #     print(name, ':', parameters, parameters.size())


        return float(loss)



    def explaining(
        self,
        model: torch.nn.Module,
        graph,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(graph.x, dict) or isinstance(graph.edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")



        hard_edge_mask = None
        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided "
                                 f"in '{self.__class__.__name__}' for "
                                 f"node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' "
                                 f"argument in '{self.__class__.__name__}'")

            # We need to compute hard masks to properly clean up edges and
            # nodes attributions not involved during message passing:
            _, hard_edge_mask = self._get_hard_masks(model, index, graph.edge_index,
                                                     num_nodes=graph.x.size(0))
        # origin prediction
        y_hat_logits = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        y_hat = F.softmax(y_hat_logits).argmax()
        y_true = graph.y[0]

        batch = graph.batch
        edge_batch = graph.batch[graph.edge_index[0, :]]
        # z = get_embeddings(model, graph.x, graph.edge_index, graph.edge_attr, graph.batch)[-1]
        z = model.get_node_reps(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        graph_embedding = global_mean_pool(z, graph.batch)
        cat_input = self._get_inputs(z, graph.edge_index)

        _, G_i_counters = extract_counterfactual_graph_embeddings(graph, model)

        logits = self.membership_fun(cat_input, G_i_counters).view(-1)
        edge_mask = self._concrete_sample(logits)

        sparsity = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        # predicted_probilities = torch.FloatTensor([]).to(device)
        fidelities_plus = torch.FloatTensor([]).to(device)
        fidelities_minus = torch.FloatTensor([]).to(device)
        fidelities_plus_prob = torch.FloatTensor([]).to(device)
        fidelities_minus_prob = torch.FloatTensor([]).to(device)

        for i in sparsity:
            top_ratio = 1.0-i
            cause_subgraph, context_subgraph, cause_visual_mask = self.pack_explanatory_subgraph(top_ratio, graph, edge_mask)
            if i == 0.5:
                v_explanation = Explanation(x=graph.x, edge_index=graph.edge_index, edge_mask=cause_visual_mask)

            cause_gnn_logits = model(cause_subgraph.x, cause_subgraph.edge_index, cause_subgraph.edge_attr, cause_subgraph.batch)
            context_gnn_logits = model(context_subgraph.x, context_subgraph.edge_index, context_subgraph.edge_attr, context_subgraph.batch)

            fidelity_minus, fidelity_plus = self.fidelity_score_acc(y_true, y_hat, F.softmax(cause_gnn_logits).argmax(), F.softmax(context_gnn_logits).argmax())
            fidelities_plus = torch.cat([fidelities_plus, fidelity_plus],dim=-1)
            fidelities_minus = torch.cat([fidelities_minus, fidelity_minus], dim=-1)

            fidelity_minus_prob, fidelity_plus_prob = self.fidelty_probility(F.softmax(y_hat_logits)[0][y_true.long()], F.softmax(cause_gnn_logits)[0][y_true.long()], F.softmax(context_gnn_logits)[0][y_true.long()])
            fidelities_plus_prob = torch.cat([fidelities_plus_prob, fidelity_plus_prob], dim=-1)
            fidelities_minus_prob = torch.cat([fidelities_minus_prob, fidelity_minus_prob], dim=-1)

        return fidelities_plus, fidelities_minus, fidelities_plus_prob, fidelities_minus_prob, v_explanation
        # return Explanation(edge_mask=edge_mask)

    def pack_explanatory_subgraph(self, top_ratio=0.2, graph=None, imp=None, relabel=True):


        top_idx = torch.LongTensor([]).to(device)
        bottom_idx = torch.LongTensor([]).to(device)
        graph_map = graph.batch[graph.edge_index[0, :]]
        exp_subgraph = graph.clone()
        con_subgraph = graph.clone()
        exp_subgraph.y = graph.y
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0]
            Gi_n_edge = len(edge_indicator)
            if math.ceil(top_ratio * Gi_n_edge)==Gi_n_edge:
                topk = min(max(math.floor(top_ratio * Gi_n_edge), 1), Gi_n_edge)
            else:
                topk = min(max(math.ceil(top_ratio * Gi_n_edge), 1), Gi_n_edge)

            Gi_pos_edge_idx = torch.sort(imp, descending=True).indices[:topk]
            Gi_nag_edge_idx = torch.sort(imp, descending=True).indices[topk:]
            top_idx = torch.cat([top_idx, edge_indicator[Gi_pos_edge_idx]])
            bottom_idx = torch.cat([bottom_idx, edge_indicator[Gi_nag_edge_idx]])
        # retrieval properties of the explanatory subgraph

        visual_mask = torch.ones_like(imp)
        visual_mask[bottom_idx] = 0.0
        # .... the edge_attr.
        if graph.edge_attr is not None:
            exp_subgraph.edge_attr = graph.edge_attr[top_idx]
            con_subgraph.edge_attr = graph.edge_attr[bottom_idx]
        # .... the edge_index.
        exp_subgraph.edge_index = graph.edge_index[:, top_idx]
        con_subgraph.edge_index = graph.edge_index[:, bottom_idx]
        # .... the nodes.
        # exp_subgraph.x = graph.x
        if relabel:
            exp_subgraph.x, exp_subgraph.edge_index, exp_subgraph.batch, exp_subgraph.pos = self._relabel(exp_subgraph, exp_subgraph.edge_index)
            con_subgraph.x, con_subgraph.edge_index, con_subgraph.batch, con_subgraph.pos = self._relabel(con_subgraph, con_subgraph.edge_index)

        return exp_subgraph, con_subgraph, visual_mask

    def _relabel(self, g, edge_index):

        sub_nodes = torch.unique(edge_index)
        x = g.x[sub_nodes]
        batch = g.batch[sub_nodes]
        row, col = edge_index
        pos = None
        try:
            pos = g.pos[sub_nodes]
        except:
            pass

        # remapping the nodes in the explanatory subgraph to new ids.
        node_idx = row.new_full((g.num_nodes,), -1)
        node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
        edge_index = node_idx[edge_index]
        return x, edge_index, batch, pos


    def supports(self) -> bool:
        explanation_type = self.explainer_config.explanation_type
        if explanation_type != ExplanationType.phenomenon:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"phenomenon explanations "
                          f"got (`explanation_type={explanation_type.value}`)")
            return False

        task_level = self.model_config.task_level
        if task_level not in {ModelTaskLevel.node, ModelTaskLevel.graph}:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"node-level or graph-level explanations "
                          f"got (`task_level={task_level.value}`)")
            return False

        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type is not None:
            logging.error(f"'{self.__class__.__name__}' does not support "
                          f"explaining input node features "
                          f"got (`node_mask_type={node_mask_type.value}`)")
            return False

        return True

    ###########################################################################

    def _get_inputs(self, embedding: Tensor, edge_index: Tensor,
                    index: Optional[int] = None) -> Tensor:
        zs = [embedding[edge_index[0]], embedding[edge_index[1]]]


        return torch.cat(zs, dim=-1)

    def _concrete_sample(self, logits: Tensor, temperature: float = 1) -> Tensor:
        random_noise = torch.rand(logits.size()).to(device)
        gate_inputs = torch.log2(random_noise) - torch.log2(1.0 - random_noise)
        gate_inputs = (gate_inputs + logits) / temperature + 1e-6
        s = gate_inputs.sigmoid()


        return s

    def _loss(self, cause_y_hat: Tensor, context_y_hat: Tensor, y: Tensor, cause_edge_mask: Tensor) -> Tensor:
        y = y.float()
        criterion = torch.nn.CrossEntropyLoss()
        overall_y_hat = F.softmax(cause_y_hat * (F.sigmoid(context_y_hat)))

        loss = criterion(overall_y_hat, y)
        # loss = criterion(cause_y_hat, y)

        # Regularization loss:
        cause_mask = cause_edge_mask
        cause_size_loss = cause_mask.sum() * self.coeffs['edge_size']

        cause_mask = 0.99 * cause_mask + 0.005
        cause_mask_ent = -cause_mask * cause_mask.log() - (1 - cause_mask) * (1 - cause_mask).log()
        cause_mask_ent_loss = cause_mask_ent.mean() * self.coeffs['edge_ent']

        # return loss + 0.1 * context_loss + self.coeffs['Regularation_loss_weight'] * (cause_size_loss + cause_mask_ent_loss)
        return loss + self.coeffs['Regularation_loss_weight'] * (cause_size_loss + cause_mask_ent_loss)


    def clip(self, x, min_val=0, max_val=1):
        return x.clamp(min_val, max_val)

    def fidelity_score_acc(self, y, y_hat, y_hat_k, y_hat_1_minus_k):
        # Binary indicator function
        I = lambda y_true, y_pred: (y_true == y_pred).float()

        # Fidelity F_-
        F_minus = I(y, y_hat) - I(y, y_hat_k)

        # Fidelity F_+
        F_plus = I(y, y_hat) - I(y, y_hat_1_minus_k)

        return F_minus.unsqueeze(0), F_plus.unsqueeze(0)

    def fidelty_probility(self, y_hat, y_hat_k, y_hat_1_minus_k):
        F_plus_pro = y_hat - y_hat_1_minus_k
        F_minus = y_hat - y_hat_k
        return F_minus.unsqueeze(0), F_plus_pro.unsqueeze(0)

# if __name__ == '__main__':
#     Train_explainer()