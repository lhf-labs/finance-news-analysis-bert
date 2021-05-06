import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sacremoses import MosesTokenizer
from collections import Counter
from itertools import chain

plt.rcParams.update({
    "font.family": "CMU Serif"
})


def compute():
    mt = MosesTokenizer(lang='en')
    df = pd.read_csv('../data/2020-08-11to2021-02-11_deduped.csv')['EVENT_TEXT']
    with Pool(18) as p:
        data = p.map(mt.tokenize, df.values)
    data = Counter(chain.from_iterable(data))
    x = list(data.keys())
    y = list(data.values())
    sort_idx = np.argsort(y)[::-1]
    x = np.take(x, sort_idx).tolist()
    y = np.take(y, sort_idx).tolist()
    with open("../output/word_data.json", 'w') as f:
        json.dump([x, y], f)


def plot():
    with open("../output/word_data.json", 'r') as f:
        data = json.load(f)
    x = data[0]
    y = data[1]
    print(f"Count {len(y)}, Mean {float(np.mean(y)):0.3f}, Std {float(np.std(y)):0.3f}, Min {np.min(y)}, Max {np.max(y)}")
    print("Top 20")
    print(x[:20])
    fig = plt.figure(figsize=(9, 4))
    fig.suptitle("Word count distribution (6m)")
    plt.xlabel("Word")
    plt.ylabel("Count (log)")
    plt.bar(range(len(x)), height=y, width=0.05, color='#30B6C2', log=True)
    fig.savefig('../output/word_data_6m.pdf')


if __name__ == '__main__':
    compute()
    plot()
