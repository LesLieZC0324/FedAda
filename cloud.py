"""
1. Server initialization
2. Server reveives updates from the user
3. Server send the aggregated information back to clients
"""

import torch
import copy
import numpy as np
import math
from utils import average_weights


class Cloud(object):
    def __init__(self, shared_layers, total, global_model, args):
        self.args = args
        self.total = total
        self.global_model = global_model
        self.receive_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.L = 0.0
        self.sigma = 0.0
        self.u = 1.0 / args.num_local_update
        self.k1 = args.num_local_update
        self.k2 = args.num_edge_aggregation
        self.size = args.num_edges
        self.communicate_power = [0.0] * self.size  # edge communication power
        self.t = [0.0] * self.size
        self.edge_max_time = [0.0] * self.size
        self.cloud_sync_time = [0.0] * self.size
        self.cloud_waiting_time = 0.0
        self.k2_list = [args.num_edge_aggregation] * self.size
        self.alpha_list = [1 / self.size] * self.size
        self.communication_overhead = [self.total * 4] * self.size
        self.init_loss = 0.0
        self.max_time = 0.0
        self.bandwidth = [0.0] * self.size

    def GenerateComm(self, edges, epoch):
        comm_kernel_val = 4.0  # edge communication average power
        comm_low = comm_kernel_val * (1 - self.args.comm_bias)
        comm_low = comm_low if comm_low > 0 else 0.1
        self.bandwidth = np.random.uniform(low=comm_low, high=comm_kernel_val * (1 + self.args.comm_bias), size=self.size)
        for i in range(self.size):
            self.communicate_power[i] = self.total * 32 / self.bandwidth[i]
            self.edge_max_time[i] = edges[i].max_time
            self.t[i] = self.k2 * self.edge_max_time[i] + self.communicate_power[i]
        return None

    def calculation(self, epoch):
        # FedAda
        if self.args.algorithm == 2:
            self.k2_list = [self.k2] * self.size
            t_min = min(self.t)
            for i in range(self.size):
                if self.t[i] != t_min:
                    temp = int(math.floor((t_min - self.communicate_power[i]) / self.edge_max_time[i]))
                    self.k2_list[i] = temp if temp > 1 else 1
                k2_sum = sum(self.k2_list)
                self.alpha_list[i] = self.k2_list[i] / k2_sum
                self.t[i] = self.k2_list[i] * self.edge_max_time[i] + self.communicate_power[i]
            self.max_time = max(self.t)
            self.cloud_waiting_time = self.max_time - t_min
            for i in range(self.size):
                self.cloud_sync_time[i] = self.max_time - self.t[i]
        # RAF
        elif self.args.algorithm == 1:
            self.k2_list = [self.k2] * self.size
            t_max = max(self.t)
            for i in range(self.size):
                if self.t[i] != t_max:
                    self.k2_list[i] = int(self.k2 * math.floor((t_max - self.communicate_power[i]) / self.edge_max_time[i]))
                self.t[i] = self.k2_list[i] * self.edge_max_time[i] + self.communicate_power[i]
            self.max_time = t_max
            self.cloud_waiting_time = self.max_time - min(self.t)
            for i in range(self.size):
                self.cloud_sync_time[i] = self.max_time - self.t[i]
        # RAF with fixed k2
        elif self.args.algorithm == 3:
            self.k2_list = [self.k2] * self.size
            t_max = max(self.t)
            for i in range(self.size):
                if self.t[i] != t_max:
                    temp = int(self.k2 * math.floor((t_max - self.communicate_power[i]) / self.edge_max_time[i]))
                    self.k2_list[i] = temp if temp < 6 else 6
                self.t[i] = self.k2_list[i] * self.edge_max_time[i] + self.communicate_power[i]
            self.max_time = t_max
            self.cloud_waiting_time = self.max_time - min(self.t)
            for i in range(self.size):
                self.cloud_sync_time[i] = self.max_time - self.t[i]
        # HierFAVG or HFL
        else:
            self.max_time = max(self.t)
            self.cloud_waiting_time = self.max_time - min(self.t)
            for i in range(self.size):
                self.cloud_sync_time[i] = self.max_time - self.t[i]
        print("This round: max_k1 = {}, k2 = {}".format(self.k1, self.k2_list))
        return sum(self.cloud_sync_time), self.k2_list, self.cloud_waiting_time

    def refresh_cloud(self):
        self.receive_buffer.clear()
        # del self.id_registration[:]
        # self.sample_registration.clear()
        return None

    def edge_registration(self, edge):
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_train_sample_num
        return None

    def receive_from_edge(self, edge_id, edge_shared_state_dict, epoch, L, sigma):
        self.receive_buffer[edge_id] = edge_shared_state_dict
        self.L += L
        self.sigma += sigma
        return None

    def aggregate(self, epoch, global_loss):
        if epoch == 0:
            self.init_loss = global_loss
        receive_dict = [edge_dict for edge_dict in self.receive_buffer.values()]
        sample_num = [edge_num for edge_num in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w=receive_dict, s_num=sample_num, alpha=self.alpha_list)
        if self.args.algorithm == 2 and epoch > 0:
            self.k1, self.u, self.k2 = self.FedAda()
        return self.k1, self.u, self.L, self.sigma

    def send_to_edge(self, edge, index):
        edge.receive_from_cloud(copy.deepcopy(self.shared_state_dict), k1=self.k1)
        self.L = 0.0
        self.sigma = 0.0
        return None

    def send_to_edge_k2(self, edge, index):
        edge.receive_from_cloud_k2(k2=self.k2_list[index])

    def FedAda(self):
        minVal = float("inf")
        self.L /= max(int(self.args.frac * self.args.num_clients), 1)
        self.sigma /= max(int(self.args.frac * self.args.num_clients), 1)
        print("L = {}, sigma = {}".format(self.L, self.sigma))
        for k in range(1, self.args.num_local_update + 1):
            for u in range(10, 101, 5):
                z = (2 * self.L * self.init_loss) / (((u / 100) ** 2) * (k ** 3) * np.sqrt(
                    (u / 100) * self.args.epochs * self.args.num_clients * self.args.frac)) \
                    + (np.sqrt((u / 100) ** 3) * k * self.sigma) / np.sqrt(
                    self.args.epochs * self.args.num_clients * self.args.frac) \
                    + (((u / 100) ** 3) * (k ** 2) * self.sigma) / (
                    self.args.epochs * self.args.num_clients * self.args.frac) \
                    + ((u / 100) * self.sigma) / self.args.epochs
                if z < minVal:
                    minVal = z
                    self.k1 = k
                    self.u = u / 100
        temp = int(round(self.k1 * self.u))
        self.k2 = temp if temp > 0 else 1
        print("Next round: max_k1 = {}, max_k2 = {}".format(self.k1, self.k2))
        return self.k1, self.u, self.k2
