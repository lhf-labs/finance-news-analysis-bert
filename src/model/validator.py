import logging


def validate(model, device, validation_data, criterion):
    """
    Validates a given model.
    """
    model.eval()
    loss_validation = 0.0
    batch_number = 0
    for idx, data in enumerate(validation_data):
        batch_number = idx + 1
        logging.info(f'Validation {batch_number}/{len(validation_data)} batches')
        # Get data
        texts, labels = data[0], data[1].float().to(device)

        # Reset model's gradient
        model.zero_grad()

        # Forward model
        outputs = model(texts)

        loss = criterion(outputs, labels)
        loss_validation += loss.item()
    return loss_validation/batch_number
