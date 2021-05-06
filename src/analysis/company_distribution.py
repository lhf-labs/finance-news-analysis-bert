import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "CMU Serif"
})


def compute():
    group = pd.read_csv('../data/2021-02-12to2021-02-26_deduped.csv').groupby("ENTITY_NAME")
    data = group.size().reset_index().values.tolist()
    x = list(map(lambda x: x[0], data))
    y = list(map(lambda y: y[1], data))
    sort_idx = np.argsort(y)[::-1]
    x = np.take(x, sort_idx).tolist()
    y = np.take(y, sort_idx).tolist()
    with open("../output/company_data.json", 'w') as f:
        json.dump([x, y], f)


def plot():
    with open("../output/company_data.json", 'r') as f:
        data = json.load(f)
    x = data[0]
    y = data[1]
    print(f"Count {len(y)}, Mean {float(np.mean(y)):0.3f}, Std {float(np.std(y)):0.3f}, Min {np.min(y)}, Max {np.max(y)}")
    print("Top 20")
    print(x[:20])
    fig = plt.figure(figsize=(9, 4))
    fig.suptitle("Company count distribution (test)")
    plt.xlabel("Company")
    plt.ylabel("Count (log)")
    plt.bar(range(len(x)), height=y, width=0.05, color='#30B6C2', log=True)
    fig.savefig('../output/company_data_test.pdf')



if __name__ == '__main__':
    compute()
    plot()
