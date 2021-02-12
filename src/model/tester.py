import logging
import numpy as np
from sklearn.metrics import accuracy_score


def test(model, dataset, device, test_data, criterion):
    """
    Tests a given model.
    """
    model.eval()
    loss_test = 0.0
    batch_number = 0
    all_labels = list()
    all_outputs = list()
    for idx, data in enumerate(test_data):
        batch_number = idx + 1
        logging.info(f'Test {batch_number}/{len(test_data)} batches')
        # Get data
        texts, labels = data[0], data[1].to(device)

        # Reset model's gradient
        model.zero_grad()

        # Forward model
        outputs = model(texts)

        loss = criterion(outputs, labels)
        loss_test += loss.item()
        all_labels.extend(labels.tolist())
        all_outputs.extend(outputs.tolist())
    logging.info(f'Test loss: {loss_test:5f}, {loss_test/batch_number:5f}')
    if dataset == 'phrasebank':
        all_outputs = np.argmax(np.array(all_outputs), axis=1)
        logging.info(f'Test accuracy score: {accuracy_score(all_labels, all_outputs)}')
    return loss_test/batch_number
