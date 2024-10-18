"""
Used to initialize the model in the client
Include initialization, training for one iteration and test function
"""

from models.mnistLR import mnistLR
from models.mnistCNN import mnistCNN
from models.LeNet import LeNet
from models.cifarCNN import cifarCNN
from models.cifarResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.mobileNet import MobileNet
from models.AlexNet import AlexNet
from models.ResNet9 import ResNet9
import torch
from torch import nn
import torch.optim as optim


class MyModel(nn.Module):
    def __init__(self, shared_layers, specific_layers, lr, momentum, args):
        super(MyModel, self).__init__()
        self.shared_layers = shared_layers
        self.specific_layers = specific_layers
        self.lr = lr
        self.lr_decay = 0.99
        self.momentum = momentum
        param_dict = [{"params": self.shared_layers.parameters()}]
        if self.specific_layers:
            param_dict += [{"params": self.specific_layers.parameters()}]
        self.optimizer = optim.SGD(params=param_dict, lr=lr, momentum=momentum, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99, last_epoch=-1)
        self.optimizer_state_dict = self.optimizer.state_dict()
        self.criterion = nn.CrossEntropyLoss()

    def optimize_model(self, input_batch, label_batch, global_model):
        self.shared_layers.train(True)
        output_batch = self.shared_layers(input_batch)
        # if self.specific_layers:
        #     self.specific_layers.train(True)
        # if self.specific_layers:
        #     output_batch = self.specific_layers(self.shared_layers(input_batch))
        # else:
        #     output_batch = self.shared_layers(input_batch)

        global_model.zero_grad()
        global_model_output = global_model(input_batch)
        global_model_loss = self.criterion(global_model_output, label_batch)
        global_batch_grad = torch.autograd.grad(global_model_loss, global_model.parameters(), retain_graph=True)

        self.optimizer.zero_grad()
        batch_loss = self.criterion(output_batch, label_batch)
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss.item(), global_batch_grad

    def model_parameters_calculation(self, inputs, labels, local_model, global_model):
        local_model.zero_grad()
        global_model.zero_grad()
        local_model_output = local_model(inputs)
        global_model_output = global_model(inputs)

        local_model_loss = self.criterion(local_model_output, labels)
        global_model_loss = self.criterion(global_model_output, labels)

        local_grad = torch.autograd.grad(local_model_loss, local_model.parameters(), retain_graph=True)
        global_grad = torch.autograd.grad(global_model_loss, global_model.parameters(), retain_graph=True)

        return local_grad, global_grad

    def test_model(self, input_batch):
        self.shared_layers.eval()
        with torch.no_grad():
            if self.specific_layers:
                output_batch = self.specific_layers(self.shared_layers(input_batch))
            else:
                output_batch = self.shared_layers(input_batch)
        return output_batch

    def update_model(self, new_shared_layers):
        self.shared_layers.load_state_dict(new_shared_layers)


def initialize_model(args, device):
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        if args.model == 'mnistLR' or args.model == 'fmnistLR':
            shared_layers = mnistLR(input_channels=1, output_channels=10)
            specific_layers = None
        elif args.model == 'mnistCNN' or args.model == 'fmnistCNN':
            shared_layers = mnistCNN(input_channels=1, output_channels=10)
            specific_layers = None
        elif args.model == 'MobileNet':
            shared_layers = MobileNet(input_channels=1, output_channels=10)
            specific_layers = None
        elif args.model == 'LeNet':
            shared_layers = LeNet(input_channels=1, output_channels=10)
            specific_layers = None
        else:
            raise ValueError('Model not implemented for MNIST/FMNIST')
    elif args.dataset == 'cifar10' or args.dataset == 'cifar':
        if args.model == 'cifarCNN':
            shared_layers = cifarCNN(input_channels=3, output_channels=10)
            specific_layers = None
        elif args.model == 'AlexNet':
            shared_layers = AlexNet(input_channels=3, output_channels=10)
            specific_layers = None
        elif args.model == 'ResNet9':
            shared_layers = ResNet9(input_channels=3, output_channels=10)
            specific_layers = None
        elif args.model == 'ResNet18':
            shared_layers = ResNet18(input_channels=3, output_channels=10)
            specific_layers = None
        else:
            raise ValueError('Model not implemented for CIFAR-10')
    elif args.dataset == 'cifar100':
        if args.model == 'ResNet9':
            shared_layers = ResNet9(input_channels=3, output_channels=100)
            specific_layers = None
        elif args.model == 'ResNet18':
            shared_layers = ResNet18(input_channels=3, output_channels=100)
            specific_layers = None
        else:
            raise ValueError('Model not implemented for CIFAR-10')
    else:
        raise ValueError('Wrong input for dataset, only mnist, fmnist and cifar10 are valid')
    if args.cuda:
        shared_layers = shared_layers.cuda(device)
    model = MyModel(shared_layers=shared_layers, specific_layers=specific_layers, lr=args.lr, momentum=args.momentum, args=args)
    return model
