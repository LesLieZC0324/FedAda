import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()

    # dataset and models
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='name of the dataset: mnist, fmnist or cifar10')
    parser.add_argument('--model', type=str, default='mnistCNN',
                        help='mnist/fmnist: mnistLR, mnistCNN; cifar10: cifarCNN, cifarResNet, mobileNet')
    parser.add_argument('--input_channels', type=int, default=1,
                        help='input channels. mnist:1, fmnist:1, cifar10 :3')
    parser.add_argument('--output_channels', type=int, default=10,
                        help='output channels')

    # training parameter
    parser.add_argument('--algorithm', type=int, default=0,
                        help='0=HierFAVG, 1=RAF, 2=FedAda')
    parser.add_argument('--compute_bias', type=float, default=0.0)
    parser.add_argument('--comm_control', type=float, default=1.0,
                        help='communication control weight, 1 or 10')
    parser.add_argument('--comm_bias', type=float, default=0.0)
    parser.add_argument('--percentage', type=float, default=0.6,
                        help='the percentage of Non-IID data')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size when training on the clients')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of rounds of training')
    parser.add_argument('--num_local_update', type=int, default=10,
                        help='number of local update (client)')
    parser.add_argument('--num_edge_aggregation', type=int, default=1,
                        help='number of edge aggregation (edge server)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate of the SGD when trained on client')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum')

    # setting for federated learning
    parser.add_argument('--iid', type=int, default=1,
                        help='distribution of the data')
    parser.add_argument('--unequal', type=int, default=0,
                        help='distribution equal or unequal of the data')
    parser.add_argument('--num_clients', type=int, default=100,
                        help='number of all available clients')
    parser.add_argument('--frac', type=float, default=0.1,
                        help='fraction of participated clients')
    parser.add_argument('--num_edges', type=int, default=2,
                        help='number of edge servers')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (defaul: 42)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to be selected, 0, 1, 2, 3')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args
