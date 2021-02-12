import math
import numpy as np
import contextlib
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import linecache
import csv
from itertools import takewhile, repeat

TEMP_SEED = 0


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def load(path, batch_size, ratios):
    dataset = FinanceDataset(path, batch_size)
    train_indexes, validation_indexes, test_indexes = split(finance_dataset=dataset, ratios=ratios)
    return DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indexes)), \
           DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(validation_indexes)),\
           DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indexes))


def split(finance_dataset, ratios):
    finance_dataset_len = len(finance_dataset)
    indexes = np.array(range(finance_dataset_len))
    with temp_seed(TEMP_SEED):
        np.random.shuffle(indexes)

    train = indexes[:math.ceil(ratios[0] * finance_dataset_len)]
    validation = indexes[
                 math.ceil(ratios[0] * finance_dataset_len):math.ceil((ratios[0] + ratios[1]) * finance_dataset_len)]
    test = indexes[math.ceil((ratios[0] + ratios[1]) * finance_dataset_len):]
    return train, validation, test


class FinanceDataset(Dataset):
    class_dict = {"negative": 0, "neutral": 1, "positive": 2}

    def __init__(self, path, chunk_size):
        super(FinanceDataset).__init__()
        self.path = path
        self.chunk_size = chunk_size
        self.len = self.count_lines(path)

    def __getitem__(self, index):
        line = linecache.getline(self.path, index + 1)
        csv_line = next(csv.reader([line], delimiter='@'))
        return csv_line[0], FinanceDataset.class_dict[csv_line[1]]

    def __len__(self):
        return self.len

    @staticmethod
    def count_lines(path):
        # https://stackoverflow.com/questions/19001402/how-to-count-the-total-number-of-lines-in-a-text-file-using-python
        f = open(path, 'rb')
        buf_gen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
        return sum(buf.count(b'\n') for buf in buf_gen if buf)
