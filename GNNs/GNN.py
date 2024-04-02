import sys
import time
import random
import argparse
import os
import os.path as osp
import numpy as np

import torch
from torch import nn
from torch.nn import ModuleList, CrossEntropyLoss
from torch.nn import Sequential as Seq, ReLU, Linear as Lin, Softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import global_mean_pool, GINEConv, BatchNorm, GraphConv
from torch_geometric.data import DataLoader
import torch.nn.functional as F

# from datasets.mutag_dataset import Mutagenicity
from torch_geometric.datasets import MoleculeNet

from Utils.pytorchtools import EarlyStopping

sys.path.append('..')
from functools import wraps


def parse_args():
    parser = argparse.ArgumentParser(description="Train BBBP Model")

    parser.add_argument('--data_path', nargs='?', default=osp.join(osp.dirname(__file__), '..', 'data', 'BBBP'),
                        help='Input data path.')
    parser.add_argument('--model_path', nargs='?', default=osp.join(osp.dirname(__file__), '..', 'checkpoint', 'gnns'),
                        help='path for saving trained model.')
    parser.add_argument('--dataset_name', nargs='?', default='BBBP',
                        help='name of dataset.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='GPU device.')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epoch.')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Interval of evaluation.')
    parser.add_argument('--num_unit', type=int, default=3,
                        help='number of Convolution layers(units)')
    parser.add_argument('--random_label', type=bool, default=False,
                        help='train a model under label randomization for sanity check')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def overload(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        if len(args) +  len(kargs) == 2:
            if len(args) == 2: # for inputs like model(g)
                g = args[1]
            else:# for inputs like model(graph=g)
                g = kargs['graph']
            return func(args[0], g.x, g.edge_index, g.edge_attr, g.batch)

        elif len(args) + len(kargs) == 5:
            if len(args) == 5: # for inputs like model(x, ..., batch)
                return func(*args)
            else: # for inputs like model(x=x, ..., batch=batch)
                return func(args[0], **kargs)

        elif len(args) +  len(kargs) == 6:
            if len(args) == 6: # for inputs like model(x, ..., batch, pos)
                return func(*args[:-1])
            else: # for inputs like model(x=x, ..., batch=batch, pos=pos)
                return func(args[0], kargs['x'], kargs['edge_index'], kargs['edge_attr'], kargs['batch'])
        else:
            raise TypeError
    return wrapper


class GNN_Net(torch.nn.Module):
    def __init__(self, conv_unit=2):
        super(GNN_Net, self).__init__()

        self.node_emb = Lin(9, 64)
        self.edge_emb = Lin(3, 1)
        self.relu_nn = ModuleList([ReLU() for i in range(conv_unit)])

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()

        conv_dim = [64, 128, 64]
        self.gnn_outputdim = 32

        for i in range(conv_unit):
            if i+1 < conv_unit:
                conv = GraphConv(conv_dim[i], conv_dim[i+1])
                self.convs.append(conv)
                self.batch_norms.append(BatchNorm(conv_dim[i+1]))
                self.relus.append(ReLU())
            else:
                conv = GraphConv(conv_dim[i], self.gnn_outputdim)
                self.convs.append(conv)
                self.batch_norms.append(BatchNorm(self.gnn_outputdim))
                self.relus.append(ReLU())

        self.lin1 = Lin(32, 16)
        self.relu = ReLU()
        self.lin2 = Lin(16, 2)
        self.softmax = Softmax(dim=1)

    @overload
    def forward(self, x, edge_index, edge_attr, batch):
        x = x.float()
        edge_attr = edge_attr.float()
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return self.get_pred(graph_x)

    @overload
    def get_node_reps(self, x, edge_index, edge_attr, batch):
        x = x.float()
        edge_attr = edge_attr.float()
        x = self.node_emb(x)
        # x = F.dropout(x, p=0.4)
        edge_attr = self.edge_emb(edge_attr)
        for conv, batch_norm, ReLU in zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_attr)
            x = ReLU(batch_norm(x))
        # x = F.dropout(x, p=0.4)
        node_x = x
        return node_x

    @overload
    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        x = x.float()
        edge_attr = edge_attr.float()
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_pred(self, graph_x):
        pred = self.relu(self.lin1(graph_x))
        pred = self.lin2(pred)
        self.readout = self.softmax(pred)
        return pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)

    def get_GNN_output_dim(self):
        return self.gnn_outputdim


def Gtrain(train_loader,
           model,
           optimizer,
           device,
           criterion=nn.MSELoss()
           ):
    model.train()
    loss_all = 0
    criterion = criterion

    for data in train_loader:
        data.to(device)
        optimizer.zero_grad()
        out = model(data.x,
                    data.edge_index,
                    data.edge_attr,
                    data.batch)
        loss = criterion(out, F.one_hot(data.y.long(), num_classes=2).squeeze(1).float())
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_loader.dataset)


def Gtest(test_loader,
          model,
          device,
          criterion=nn.L1Loss(reduction='mean'),
          ):

    model.eval()
    error = 0
    correct = 0

    with torch.no_grad():

        for data in test_loader:
            data = data.to(device)
            output = model(data.x,
                           data.edge_index,
                           data.edge_attr,
                           data.batch,
                           )

            error += criterion(output, F.one_hot(data.y.long(), num_classes=2).squeeze(1).float()) * data.num_graphs
            correct += float(output.argmax(dim=1).eq(data.y.squeeze(1).long()).sum().item())

        return error / len(test_loader.dataset), correct / len(test_loader.dataset)


if __name__ == '__main__':

    set_seed(0)
    args = parse_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    dataset = MoleculeNet('../data', 'BBBP')
    torch.manual_seed(42)
    data_random = dataset.shuffle()
    filtered_data_list = [data for data in data_random if data.num_nodes > 0]

    train_dataset = filtered_data_list[201:]
    val_dataset = filtered_data_list[:100]
    test_dataset = filtered_data_list[101:200]

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False
                             )
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False
                            )
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True
                              )
    model = GNN_Net(args.num_unit).to(device)

    save_path = args.dataset_name + '_gcn_model.pth'
    if not osp.exists(args.model_path):
        os.makedirs(args.model_path)
    model_save_path = osp.join(args.model_path, save_path)
    early_stopping = EarlyStopping(200, verbose=True, path=model_save_path)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr
                                 )
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.8,
                                  patience=10,
                                  min_lr=1e-4
                                  )
    min_error = None
    for epoch in range(1, args.epoch + 1):

        t1 = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']

        loss = Gtrain(train_loader,
                      model,
                      optimizer,
                      device=device,
                      criterion=CrossEntropyLoss()
                      )

        _, train_acc = Gtest(train_loader,
                             model,
                             device=device,
                             criterion=CrossEntropyLoss()
                             )

        val_error, val_acc = Gtest(val_loader,
                                   model,
                                   device=device,
                                   criterion=CrossEntropyLoss()
                                   )
        test_error, test_acc = Gtest(test_loader,
                                     model,
                                     device=device,
                                     criterion=CrossEntropyLoss()
                                     )
        scheduler.step(val_error)


        early_stopping(val_error, model)

        if min_error is None or val_error <= min_error:
            min_error = val_error

        t2 = time.time()

        if epoch % args.verbose == 0:
            test_error, test_acc = Gtest(test_loader,
                                         model,
                                         device=device,
                                         criterion=CrossEntropyLoss()
                                         )
            t3 = time.time()
            print('Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Test Loss: {:.5f}, '
                  'Test acc: {:.5f}'.format(epoch, t3 - t1, lr, loss, test_error, test_acc))
            continue

        print('Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Train acc: {:.5f}, Validation Loss: {:.5f}, '
              'Validation acc: {:5f}'.format(epoch, t2 - t1, lr, loss, train_acc, val_error, val_acc))
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            break

