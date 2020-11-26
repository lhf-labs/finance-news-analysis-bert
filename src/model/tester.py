import logging


def test(model, device, test_data, criterion):
    """
    Tests a given model.
    """
    model.eval()
    loss_test = 0.0
    batch_number = 0
    for idx, data in enumerate(test_data):
        batch_number = idx + 1
        logging.info(f'Test {batch_number}/{len(test_data)} batches')
        # Get data
        texts, labels = data[0], data[1].float().to(device)

        # Reset model's gradient
        model.zero_grad()

        # Forward model
        outputs = model(texts)

        loss = criterion(outputs, labels)
        loss_test += loss.item()
    return loss_test/batch_number
