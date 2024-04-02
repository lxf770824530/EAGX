import random
import time
import copy
import math
import torch
import torch.nn.functional as F


import os
from config import arg_parse
args = arg_parse()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = args.device


class Fcm_for_GNNX(object):
    def __init__(self):
        # self.edge_features = _Edge_features
        # self.global_feature = _Global_feature

        self.Epsilon = 0.001      # 0.001
        self.MAX = 1000             # 1000
        self.cluster_num = 2
        self.m = 2
        self.alpha = 0.3         # 0.1




    def randomize_data(self, data):
        """
        该功能将数据随机化，并保持随机化顺序的记录
        """
        order = list(range(0, len(data)))
        random.shuffle(order)
        new_data = [[] for i in range(0, len(data))]
        for index in range(0, len(order)):
            new_data[index] = data[order[index]]
        return new_data, order

    def initialize_U(self, data, cluster_number):

        n_data_points = data.size(0)

        # 随机生成一个n_data_points x n_clusters的矩阵
        U = torch.rand(n_data_points, cluster_number)

        # 归一化每行，使得每行的和为1
        U = U / U.sum(dim=1, keepdim=True)

        return U


    def end_condition(self, U, U_old):
        """
    	结束条件。当U矩阵随着连续迭代停止变化时，触发结束
    	"""
        dv = torch.abs(U-U_old)
        max_val = torch.max(dv)
        if max_val > self.Epsilon:
            return False
        else:
            return True
        # for i in range(0, len(U)):
        #     for j in range(0, len(U[0])):
        #         if abs(U[i][j] - U_old[i][j]) > self.Epsilon:
        #             return False
        # return True

    def normalise_U(self,U):
        """
        在聚类结束时使U模糊化。每个样本的隶属度最大的为1，其余为0
        """
        U_ = torch.argmax(U,dim=1)
        U_ = F.one_hot(U_, num_classes=self.cluster_num)
        return U_

    def fuzzy(self, data, graph_embedding):
        """
        这是主函数，它将计算所需的聚类中心，并返回最终的归一化隶属矩阵U.
        参数是：簇数(cluster_number)和隶属度的因子(m)
        """
        # 初始化隶属度矩阵U
        U = self.initialize_U(data, self.cluster_num)
        count = 0
        # 循环更新U
        # print('Extracting fuzzy features......')

        for _ in range(15):
            count += 1

            # 创建隶属度矩阵U的深度拷贝，以便稍后进行比较
            U_old = U.clone().to(device)

            # 根据当前的隶属度矩阵U计算聚类中心C

            weights = U.pow(self.m).to(device)

            #计算所有边的的聚类中心
            edges_C_numerator_con = ((-1) ** torch.arange(self.cluster_num).view(-1, 1)).to(device) * self.alpha * graph_embedding.view(1,-1)
            u_Embedding = weights.t().view(self.cluster_num, -1, 1) * data
            sum_u_Embedding = torch.sum(u_Embedding, dim=1)
            edges_C_numerator = edges_C_numerator_con + sum_u_Embedding

            edges_C_denominator_con = ((-1) ** torch.arange(self.cluster_num)).float().view(-1, 1).to(device) * self.alpha
            sum_u = torch.sum(weights, dim=0)
            edges_C_denominator = edges_C_denominator_con.squeeze(1) + sum_u
            edges_C = edges_C_numerator / edges_C_denominator.view(-1,1)

            #计算所有边的隶属度矩阵
            distances = torch.norm(data[:, None, :] - edges_C, dim=2)
            distances_pop = distances.pow(2/(1-self.m))
            row_sum = torch.sum(distances_pop, dim=1).view(-1, 1)
            edges_U = distances_pop / row_sum
            U = edges_U

            # 检查是否满足结束条件，如果满足则跳出循环
            if self.end_condition(U, U_old):
                break

        return U, edges_C, count

    def Run_fcm(self, edge_embedding, graph_embedding):

        # start = time.time()
        final_location, centers, cycled_count = self.fuzzy(edge_embedding, graph_embedding)

        # print("Time consumption：{0}".format(time.time() - start))
        # print("The update number of U：", cycled_count)
        # print(final_location)
        final_location_normalised = self.normalise_U(final_location)
        # print(final_location_normalised)
        important_edge_index = []
        for u_ in final_location_normalised:
            if u_ == [0,1]:
                important_edge_index.append(True)
            else:
                important_edge_index.append(False)

        return final_location, centers, important_edge_index






