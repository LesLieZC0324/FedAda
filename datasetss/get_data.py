from datasets.sampling import get_dataset


def get_dataloaders(args):
    if args.dataset in ['mnist', 'fmnist', 'cifar10', 'cifar100']:
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_dataset(dataset=args.dataset, args=args)
    else:
        raise ValueError("This dataset is not implemented yet")
    return train_loaders, test_loaders, v_train_loader, v_test_loader
