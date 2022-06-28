import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from multiprocessing import Pool, freeze_support


def set_color(_fig, _ax):
    _fig.patch.set_facecolor('#1b212c')
    _ax.patch.set_facecolor('#1b212c')
    _ax.spines['bottom'].set_color('white')
    _ax.spines['top'].set_color('white')
    _ax.spines['left'].set_color('white')
    _ax.spines['right'].set_color('white')
    _ax.xaxis.label.set_color('white')
    _ax.yaxis.label.set_color('white')
    _ax.grid(alpha=0.1)
    _ax.title.set_color('white')
    _ax.tick_params(axis='x', colors='white')
    _ax.tick_params(axis='y', colors='white')


def save_plot(path):
    fig, ax = plt.subplots(ncols=1, figsize=(10, 8), dpi=75)
    try:
        data = np.load(path)
    except ValueError:
        print(path)
        return
    xs = []
    ys = []
    color = []
    for index, frame in enumerate(data):
        for x in frame[0::2]:
            xs.append(x)
            color.append(index)
        for y in frame[1::2]:
            ys.append(1 - y)
    df = pd.DataFrame({"x": xs, "y": ys, "color": color})
    sns.scatterplot(x="x", y="y", hue="color", palette=sns.color_palette("hls", 20), data=df, ax=ax, alpha=0.1,
                    legend=False)
    set_color(fig, ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.axis('off')
    plt.savefig(f"../pictures/{path.split('/')[-2]}/{path.split('/')[-1][:-4]}.png", bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    freeze_support()
    for folder in os.listdir(os.path.join("../data")):

        # check if it is an actual folder
        if not os.path.isdir(os.path.join("../data", folder)):
            continue

        if not os.path.exists(os.path.join(f"../pictures/{folder}")):
            os.makedirs(f"../pictures/{folder}")

        paths = [os.path.join(f"../data/{folder}/{seq}") for seq in os.listdir(os.path.join("../data", folder))]
        print(paths)
        with Pool(processes=16) as pool:
            pool.map(save_plot, paths)

        print(f"{folder} done")