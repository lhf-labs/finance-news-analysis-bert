import logging
import numpy as np
from sklearn.metrics import accuracy_score


def validate(model, dataset, device, validation_data, criterion):
    """
    Validates a given model.
    """
    model.eval()
    loss_validation = 0.0
    batch_number = 0
    all_labels = list()
    all_outputs = list()
    for idx, data in enumerate(validation_data):
        batch_number = idx + 1
        logging.info(f'Validation {batch_number}/{len(validation_data)} batches')
        # Get data
        texts, labels = data[0], data[1].to(device)

        # Reset model's gradient
        model.zero_grad()

        # Forward model
        outputs = model(texts)

        loss = criterion(outputs, labels)
        loss_validation += loss.item()
        all_labels.extend(labels.tolist())
        all_outputs.extend(outputs.tolist())
    if dataset == 'phrasebank':
        all_outputs = np.argmax(np.array(all_outputs), axis=1)
        logging.info(f'Validation accuracy score: {accuracy_score(all_labels, all_outputs)}')
    return loss_validation/batch_number
