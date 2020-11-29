import os
import logging
import torch
import numpy as np
from model.validator import validate


def train(model, device, data_train, data_validation, epochs, criterion, optimizer, es_patience, experiment_dir):
    """
        Train a given model.
    """
    # Set initial parameters
    best_valid_loss = np.Inf
    patience = 0
    for epoch in range(1, epochs+1):
        logging.info(f'Epoch {epoch} |')

        # Set model on training
        model.train()
        # Reset epoch training loss
        loss_train = 0.0
        batch_number = 0
        for idx, data in enumerate(data_train):
            batch_number = idx + 1
            logging.info(f'Train {batch_number}/{len(data_train)} batches')
            # Get data
            texts, labels = data[0], data[1].float().to(device)

            # Reset model's gradient
            model.zero_grad()

            # Forward model
            outputs = model(texts)

            # Check loss
            loss = criterion(outputs, labels)

            # Backpropagate
            loss.backward()
            optimizer.step()

            # Get total loss of epoch
            loss_train += loss.item()
        logging.info(f'train: average loss = {loss_train/batch_number:.5f}')
        # Get validation loss
        loss_validation = validate(model, device, data_validation, criterion)
        logging.info(f'validation: average loss = {loss_train:.5f}')

        torch.save(model.state_dict(), os.path.join(experiment_dir, 'checkpoint_last.pt'))

        # Early stopping criterion
        if loss_validation < best_valid_loss:
            torch.save(model.state_dict(), os.path.join(experiment_dir, 'checkpoint_best.pt'))
            patience = 0
            logging.info(f'train: early stopping patience reset {patience}/{es_patience}')
        else:
            patience += 1
            logging.info(f'train: early stopping patience increment {patience}/{es_patience}')
            if patience == es_patience:
                logging.info(f'train: early stopping on epoch {epoch}')
                break
