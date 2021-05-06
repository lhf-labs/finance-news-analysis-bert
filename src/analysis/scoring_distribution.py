import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "CMU Serif"
})

plt.rcParams['axes.unicode_minus'] = False


files = [
    ['2y', '../data/2019-02-11to2021-02-11_deduped.csv'],
    ['1y', '../data/2020-02-11to2021-02-11_deduped.csv'],
    ['6m', '../data/2020-08-11to2021-02-11_deduped.csv'],
    ['test', '../data/2021-02-12to2021-02-26_deduped.csv']
]


def compute_plot():
    data = list()
    labels = list()
    for label, file in files:
        y = pd.read_csv(file)['EVENT_SENTIMENT_SCORE'].tolist()
        print(
            f"{label}: Count {len(y)}, Mean {float(np.mean(y)):0.3f}, Std {float(np.std(y)):0.3f}, Min {np.min(y)}, Max {np.max(y)}")
        data.append(y)
        labels.append(label)

    fig, ax = plt.subplots()

    fig.suptitle("Score Distributions")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Sentiment Value")

    ax.boxplot(data)
    ax.set_xticklabels(labels)

    fig.savefig('../output/score_data.pdf')


if __name__ == '__main__':
    compute_plot()
