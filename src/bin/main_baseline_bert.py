import os
import git
import time
import torch
import logging
import argparse
import numpy as np
from controller.data import load_data
from controller.model import prepare_device, prepare_preliminary, train_model, test_model
from model.baseline_bert_classifier import BaselineBERTClassifier

"""
Train and test a given model.
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Global parameters
    parser.add_argument('data_path', help="Path to data.", type=str)
    parser.add_argument('experiment_path', help="Path to experiment.", type=str)
    parser.add_argument('--experiment-name', help="Experiment name.", type=str, default='baseline_bert')

    # Model architecture parameters
    parser.add_argument('--number-layers', help="Number of layers of the model.", type=int, default=1)
    parser.add_argument('--layer-size', help="Layer start size of the model.", type=int, default=256)
    parser.add_argument('--minimum-layer-size', help="Layer minimum size of the model.", type=int, default=8)
    parser.add_argument('--dropout-rate', help="Dropout that is applied to the model.", type=int, default=0.2)

    # Model training parameters
    parser.add_argument('--seed', help="Seed to be used for randomization purposes.", type=int, default=42)
    parser.add_argument('--batch-size', help="Batch size to feed the model.", type=int, default=32)
    parser.add_argument('--epochs', help="Number of epochs to train the model.", type=int, default=20)
    parser.add_argument('--optimizer', help="Optimizer for training the model.", type=str, choices=['adam'],
                        default='adam')
    parser.add_argument('--lr', help="Learning rate for training the model.", type=float, default=0.001)
    parser.add_argument('--criterion', help="Loss function for training the model.", type=str, choices=['mse'],
                        default='mse')
    parser.add_argument('--patience', help="Patience for the Early Stopping.", type=int, default=5)
    parser.add_argument('--ratio', help="Train ratio, rest divided by two for validation and test.", type=float,
                        default=0.995)
    parser.add_argument('--no-cuda', help="Avoid using cuda.", action='store_true')
    args = parser.parse_args()

    # Experiments directory
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    ts = time.time()
    experiment_directory = os.path.join(args.experiment_path, f'{args.experiment_name}_{ts}_{sha[:7]}')
    os.makedirs(experiment_directory, exist_ok=True)

    # Logger
    logging.basicConfig(filename=os.path.join(experiment_directory, 'process.log'), level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())

    # Seeds
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.backends.cudnn.deterministic = True
    # torch.set_deterministic(True)
    np.random.seed(args.seed)

    # Load data
    data_loader_train, data_loader_valid, data_loader_test = load_data(path=args.data_path, batch_size=args.batch_size,
                                                                       ratio=args.ratio)

    # Build classifier
    device = prepare_device(args.no_cuda)
    classifier = BaselineBERTClassifier(model='bert-base-uncased', device=device, number_layers=args.number_layers,
                                        layer_size=args.layer_size, minimum_layer_size=args.minimum_layer_size,
                                        dropout_rate=args.dropout_rate)
    classifier.to(device)

    # Train and test
    criterion, optimizer = prepare_preliminary(args, classifier)
    train_model(classifier=classifier, device=device, data_loader_train=data_loader_train, epochs=args.epochs,
                data_loader_valid=data_loader_valid, criterion=criterion, optimizer=optimizer, patience=args.patience,
                experiment_directory=experiment_directory)
    test_model(model=classifier, device=device, data_loader_test=data_loader_test, criterion=criterion)
