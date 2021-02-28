from data.finance_dataset import load as load_finance
from data.finantial_phrasebank_dataset import load as load_phrasebank


def load_data(dataset, path, batch_size, ratio):
    ratios = [ratio]
    validation_test_ratios = 1 - ratio
    validation_test_ratios /= 2
    ratios.extend([validation_test_ratios, validation_test_ratios])
    if dataset == "finance":
        return load_finance(path, batch_size, ratios)
    else:
        return load_phrasebank(path, batch_size, ratios)


def load_data_test(dataset, path, batch_size):
    if dataset == "finance":
        return load_finance(path, batch_size, [0.0, 0.0, 1.0])
    else:
        return load_phrasebank(path, batch_size, [0.0, 0.0, 1.0])
