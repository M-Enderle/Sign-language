import os

from matplotlib import pyplot as plt
from utils import *
import seaborn as sns
import pandas as pd


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


for folder in os.listdir(os.path.join(DATA_PATH)):
    if not os.path.exists(os.path.join(f"../pictures/{folder}")):
        os.makedirs(f"../pictures/{folder}")

    for seq in os.listdir(os.path.join(DATA_PATH, folder)):
        print(folder, seq, sep="/")
        fig, ax = plt.subplots(ncols=1, figsize=(10, 8), dpi=300)
        data = np.load(os.path.join("../data/", str(folder), str(seq)))
        xs = []
        ys = []
        color = []
        for index, frame in enumerate(data):
            for x in frame[0::2]:
                xs.append(x)
                color.append(index)
            for y in frame[1::2]:
                ys.append(1 - y)

        # make dataframe with x and y coordinates
        df = pd.DataFrame({"x": xs, "y": ys, "color": color})
        # plot x and y as seaborn scatter plot
        sns.scatterplot(x="x", y="y", hue="color", palette=sns.color_palette("hls", 20), data=df, ax=ax, alpha=0.1, legend=False)

        set_color(fig, ax)
        # set x and y limits to 0 and 1
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.axis('off')

        # save the plot
        plt.savefig(f"../pictures/{folder}/{seq[:-4]}.png", bbox_inches='tight', pad_inches=0)
        plt.show()
