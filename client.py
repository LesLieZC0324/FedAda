"""
1. Client initialization, dataloaders, model(include optimizer)
2. Client model update
3. Client send updates to edge server
4. Client receives updates from edge server
5. Client modify local model based on the feedback from the edge server
"""

import torch
from torch import nn
from models.model_initialization import initialize_model
import copy


class Client(object):
    def __init__(self, id, train_loader, test_loader, device, global_model, args):
        self.args = args
        self.id = id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = initialize_model(args, device)
        self.receive_buffer = {}
        self.batch_size = args.batch_size
        self.epoch = 0
        self.pre_round_model = global_model
        self.global_model = global_model
        self.device = device
        self.L = 0.0
        self.sigma = 0.0
        self.u = 1.0 / args.num_local_update
        self.k1 = args.num_local_update
        self.k2 = args.num_edge_aggregation
        self.grad_diff_norm1 = []
        self.grad_diff_norm2 = []
        self.w_compute = {}

    def local_update(self, device):
        epoch_count = 0
        loss = 0.0
        global_batch_grad, local_grad, global_grad = [], [], []
        train_data, train_label = [], []
        end = False
        # the upper bound of the local_update is 1000 (never reached)
        for _ in range(1000):
            for k, data in enumerate(self.train_loader):
                inputs, labels = data
                if self.args.algorithm == 2 and self.epoch == 0 and k < 20:
                    train_data.append(inputs)
                    train_label.append(labels)
                inputs = inputs.to(device)
                labels = labels.to(device)

                batch_loss, global_batch_grad = self.model.optimize_model(input_batch=inputs, label_batch=labels,
                                                                          global_model=self.global_model)
                loss += batch_loss
                epoch_count += 1
                if epoch_count >= self.k1:
                    end = True
                    break
            if end:
                # self.model.scheduler.step()
                break

        if self.args.algorithm == 2 and self.epoch == 0:
            train_data = torch.cat(train_data, dim=0)
            train_label = torch.cat(train_label, dim=0)
            train_data = train_data.to(device)
            train_label = train_label.to(device)
            local_grad, global_grad = self.model.model_parameters_calculation(
                inputs=train_data, labels=train_label, local_model=self.pre_round_model, global_model=self.global_model)

            self.grad_diff_norm1 = [grad2 - grad1 for grad1, grad2 in zip(local_grad, global_grad)]
            self.grad_diff_norm2 = [grad2 - grad1 for grad1, grad2 in zip(global_batch_grad, global_grad)]

        loss /= self.k1
        return loss

    def test_model(self, device):
        acc = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model.test_model(input_batch=inputs)
                _, predict = torch.max(outputs, 1)
                total += labels.size(0)
                acc += (predict == labels).sum().item()
        return acc, total

    def send_to_edge(self, edge, epoch):
        if self.args.algorithm == 2:
            if epoch > 0:
                if self.epoch == 0:
                    for name, param in self.pre_round_model.state_dict().items():
                        self.w_compute[name] = param - self.global_model.state_dict()[name]
                    w_compute = torch.norm(torch.cat([tensor.flatten() for tensor in self.w_compute.values()]), p=1)
                    grad_norm1 = torch.norm(torch.cat([tensor.flatten() for tensor in self.grad_diff_norm1]), p=1)
                    self.L = float(abs(grad_norm1 / w_compute))
                    self.sigma = float(torch.norm(torch.cat([tensor.flatten() for tensor in self.grad_diff_norm2]), p=2))
                    # if self.id == 1:
                    #     print("grad={}, w_compute={}".format(grad_norm1, w_compute))
                    #     print("sigma={}".format(self.sigma))

        edge.receive_from_client(client_id=self.id,
                                 client_shared_state_dict=copy.deepcopy(self.model.shared_layers.state_dict()),
                                 epoch=epoch, L=self.L, sigma=self.sigma)
        self.epoch += 1
        if self.epoch == self.k2:
            self.pre_round_model = copy.deepcopy(self.model.shared_layers)  # record the final local model
            self.epoch = 0
        return None

    def receive_from_edge(self, shared_state_dict, k1, k2):
        self.receive_buffer = shared_state_dict
        self.k1 = k1
        self.k2 = k2
        return None

    def sync_with_edge(self):
        self.model.update_model(self.receive_buffer)
        if self.epoch == 0:
            self.global_model.load_state_dict(state_dict=copy.deepcopy(self.receive_buffer))
        return None
