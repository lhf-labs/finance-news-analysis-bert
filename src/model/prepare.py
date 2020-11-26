import torch


def get_device(no_cuda):
    if no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    return device


def prepare(args, model):
    if args.criterion == 'mse':
        criterion = torch.nn.MSELoss()
    else:
        raise NotImplementedError("Criterion not implemented")

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError("Criterion not implemented")

    return criterion, optimizer
