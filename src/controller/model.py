from model.prepare import prepare, get_device
from model.trainer import train
from model.tester import test


def prepare_device(no_cuda):
    return get_device(no_cuda)


def prepare_preliminary(args, model):
    return prepare(args, model)


def train_model(classifier, dataset, device, data_loader_train, data_loader_valid, epochs, criterion, optimizer, patience,
                experiment_directory):
    train(classifier, dataset, device, data_loader_train, data_loader_valid, epochs, criterion, optimizer, patience,
          experiment_directory)


def test_model(model, dataset, device, data_loader_test, criterion):
    test(model, dataset, device, data_loader_test, criterion)
