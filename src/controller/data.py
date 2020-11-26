from data.finance_dataset import load


def load_data(path, batch_size, ratio):
    ratios = [ratio]
    validation_test_ratios = 1-ratio
    validation_test_ratios /= 2
    ratios.extend([validation_test_ratios, validation_test_ratios])
    return load(path, batch_size, ratios)
