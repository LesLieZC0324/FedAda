"""
1. Edge Server initialization
2. Edge Server receives updates from the client
3. Edge Server sends the aggregated information back to clients
4. Edge Server sends the updates to the Cloud Server
5. Edge Server receives the aggregated information from the Cloud Server
"""

import torch
import math
import copy
import numpy as np
from utils import average_weights


class Edge(object):
    def __init__(self, id, client_ids, shared_layers, total, global_model, args):
        self.args = args
        self.id = id
        self.client_ids = client_ids
        self.total = total
        self.receive_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.all_train_sample_num = 0.0
        self.shared_state_dict = shared_layers.state_dict()
        self.epoch = 0
        self.global_model = global_model
        self.L = 0.0
        self.sigma = 0.0
        self.u = 1.0 / args.num_local_update
        self.k1 = args.num_local_update
        self.k2 = args.num_edge_aggregation
        self.size = int(args.num_clients / args.num_edges * args.frac)
        self.max_time = 0.0
        self.edge_sync_time = [0.0] * self.size
        self.edge_waiting_time = 0.0
        self.compute_power = [0.0] * self.size  # client compute power
        self.communicate_power = [0.0] * self.size  # client communication power
        self.t = [0.0] * self.size
        self.k1_list = [args.num_local_update] * self.size
        self.alpha_list = [1 / self.size] * self.size
        self.communication_overhead = [self.total * 4] * self.size
        self.bandwidth = [0.0] * self.size

    def GenerateCompandComm(self, epoch):
        compute_kernel_val = 0.0  # client compute average power
        comm_kernel_val = 4.0  # client communication average power
        if self.args.dataset == 'fmnist':
            compute_kernel_val = 0.5
        elif self.args.dataset == 'cifar10':
            compute_kernel_val = 4.0
        compute_low = compute_kernel_val * (1 - self.args.compute_bias)
        comm_low = comm_kernel_val * (1 - self.args.comm_bias)
        compute_low = compute_low if compute_low > 0 else 0.01
        comm_low = comm_low if comm_low > 0 else 0.1
        self.compute_power = np.random.uniform(low=compute_low,
                                               high=compute_kernel_val * (1 + self.args.compute_bias),
                                               size=self.size)
        self.bandwidth = np.random.uniform(low=self.args.comm_control * comm_low,
                                           high=self.args.comm_control * comm_kernel_val * (1 + self.args.comm_bias),
                                           size=self.size)
        for i in range(self.size):
            self.communicate_power[i] = self.total * 32 / self.bandwidth[i]
            self.t[i] = self.k1 * self.compute_power[i] + self.communicate_power[i]
        return min(self.t)

    def calculate_function(self, epoch, min_time):
        self.communication_overhead = self.total * 4 * self.size
        # FedAda
        if self.args.algorithm == 2:
            self.k1_list = [self.k1] * self.size
            min_time = min(self.t)
            for i in range(self.size):
                if self.t[i] != min_time:
                    temp = int(math.floor((min_time - self.communicate_power[i]) / self.compute_power[i]))
                    self.k1_list[i] = temp if temp > 1 else 1
                k1_sum = sum(self.k1_list)
                self.alpha_list[i] = self.k1_list[i] / k1_sum
                self.t[i] = self.k1_list[i] * self.compute_power[i] + self.communicate_power[i]
            self.max_time = max(self.t)
            self.edge_waiting_time = self.max_time - min_time
            for i in range(self.size):
                self.edge_sync_time[i] = self.max_time - self.t[i]
        # RAF
        elif self.args.algorithm == 1:
            self.k1_list = [self.k1] * self.size
            t_max = max(self.t)
            for i in range(self.size):
                if self.t[i] != t_max:
                    self.k1_list[i] = int(math.floor((t_max - self.communicate_power[i]) / self.compute_power[i]))
                    # self.k1_list[i] = temp if temp < 20 else 20
                self.t[i] = self.k1_list[i] * self.compute_power[i] + self.communicate_power[i]
            self.max_time = t_max
            self.edge_waiting_time = self.max_time - min(self.t)
            for i in range(self.size):
                self.edge_sync_time[i] = self.max_time - self.t[i]
        # RAF with fixed k1
        elif self.args.algorithm == 3:
            self.k1_list = [self.k1] * self.size
            t_max = max(self.t)
            for i in range(self.size):
                if self.t[i] != t_max:
                    temp = int(math.floor((t_max - self.communicate_power[i]) / self.compute_power[i]))
                    self.k1_list[i] = temp if temp < 30 else 30
                self.t[i] = self.k1_list[i] * self.compute_power[i] + self.communicate_power[i]
            self.max_time = t_max
            self.edge_waiting_time = self.max_time - min(self.t)
            for i in range(self.size):
                self.edge_sync_time[i] = self.max_time - self.t[i]
        # HierFAVG or HFL
        else:
            self.max_time = max(self.t)
            self.edge_waiting_time = self.max_time - min(self.t)
            for i in range(self.size):
                self.edge_sync_time[i] = self.max_time - self.t[i]
        print(self.k1_list)
        return sum(self.edge_sync_time), self.edge_waiting_time

    def refresh_edge(self):
        self.receive_buffer.clear()
        # del self.id_registration[:]
        # self.sample_registration.clear()
        return None

    def client_register(self, client):
        self.id_registration.append(client.id)
        self.sample_registration[client.id] = len(client.train_loader.dataset)
        return None

    def receive_from_client(self, client_id, client_shared_state_dict, epoch, L, sigma):
        self.receive_buffer[client_id] = client_shared_state_dict
        self.L += (L / self.k2)
        self.sigma += (sigma / self.k2)
        return None

    def aggregate(self, args):
        receive_dict = [client_dict for client_dict in self.receive_buffer.values()]
        sample_num = [client_num for client_num in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w=receive_dict, s_num=sample_num, alpha=self.alpha_list)
        self.epoch += 1
        if self.epoch == self.k2:
            self.epoch = 0
            return True
        return False

    def send_to_client(self, client, index):
        client.receive_from_edge(copy.deepcopy(self.shared_state_dict), k1=self.k1_list[index], k2=self.k2)
        return None

    def send_to_cloud(self, cloud, epoch):
        cloud.receive_from_edge(edge_id=self.id, edge_shared_state_dict=copy.deepcopy(self.shared_state_dict),
                                epoch=epoch, L=self.L, sigma=self.sigma)
        # reset L and sigma
        self.L = 0.0
        self.sigma = 0.0
        return None

    def receive_from_cloud(self, shared_state_dict, k1):
        self.shared_state_dict = shared_state_dict
        self.k1 = k1
        return None

    def receive_from_cloud_k2(self, k2):
        self.k2 = k2
        return k2
