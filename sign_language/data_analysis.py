import os

import imageio
from matplotlib import pyplot as plt
import matplotlib.animation as animation
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


"""
for folder in os.listdir(os.path.join(DATA_PATH)):
    if not os.path.exists(os.path.join(f"../pictures/{folder}")):
        os.makedirs(f"../pictures/{folder}")

    for seq in os.listdir(os.path.join(DATA_PATH, folder)):
        if os.path.exists(os.path.join(f"../pictures/{folder}/{seq[:-4]}.png")):
            continue
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
        plt.close(fig)
"""

files = [x for x in range(10)]

for file in files:
    picture = os.path.join("../size_aug", f"2_size_{file}_aug.npy")
    data = np.load(picture)
    col_pal = sns.color_palette("Spectral", n_colors=len(data))
    for frame in range(1, len(data)):
        fig, ax = plt.subplots(ncols=1, figsize=(10, 8), dpi=75)
        xs = []
        ys = []
        for x in data[frame][0::2]:
            xs.append(x)
        for y in data[frame][1::2]:
            ys.append(1 - y)
        color = col_pal[frame]
        sns.scatterplot(x="x", y="y", color=color, data=pd.DataFrame({"x": xs, "y": ys}), ax=ax, legend=False)

        set_color(fig, ax)
        # set x and y limits to 0 and 1
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.axis('off')
        plt.savefig(f"../gif/{file}_{frame-1}.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    col_pal = sns.color_palette("Spectral", n_colors=len(data))
    fig, ax = plt.subplots(ncols=1, figsize=(10, 8), dpi=75)
    xs = []
    ys = []
    for x in data[0][0::2]:
        xs.append(x)
    for y in data[0][1::2]:
        ys.append(1 - y)
    sns.scatterplot(x="x", y="y", palette=col_pal[0], data=pd.DataFrame({"x": xs, "y": ys}), ax=ax, legend=False)
    set_color(fig, ax)
    # set x and y limits to 0 and 1
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.axis('off')
    plt.savefig(f"../gif/{file}_{len(data)-1}.png", bbox_inches='tight', pad_inches=0)


    # create gif from folder gif with images
    images = []
    for frame in range(len(data)):
        images.append(imageio.v2.imread(f"../gif/{file}_{frame}.png"))
    imageio.mimsave(f"../gif_2/{file}.gif", images, duration=0.1)
    plt.close(fig)

    for _file in os.listdir(os.path.join("../gif/")):
        if _file.endswith(".png"):
            os.remove(os.path.join("../gif/", _file))

"""
from apply_augmentation import augmentation
augmentation()

folder = os.path.join(DATA_PATH, "world_aug")

for seq in os.listdir(folder):
    fig, ax = plt.subplots(ncols=1, figsize=(10, 8), dpi=300)
    data = np.load(os.path.join(str(folder), str(seq)))
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
    sns.scatterplot(x="x", y="y", hue="color", palette=sns.color_palette("Spectral", n_colors=20), data=df, ax=ax, alpha=0.1, legend=False)

    set_color(fig, ax)
    # set x and y limits to 0 and 1
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.axis('off')

    # save the plot
    plt.savefig(f"../gif_aug/{seq[:-4]}.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)
"""
"""
for pic in os.listdir(os.path.join(DATA_PATH, "world_aug")):
    picture = os.path.join(DATA_PATH, "world_aug", pic)
    data = np.load(picture)

    col_pal = sns.color_palette("Spectral", n_colors=20)
    for frame in range(1, len(data)):
        fig, ax = plt.subplots(ncols=1, figsize=(10, 8), dpi=300)
        xs = []
        ys = []
        for x in data[frame][0::2]:
            xs.append(x)
        for y in data[frame][1::2]:
            ys.append(1 - y)

        color = col_pal[frame]
        sns.scatterplot(x="x", y="y", color=color, data=pd.DataFrame({"x": xs, "y": ys}), ax=ax, legend=False)

        set_color(fig, ax)
        # set x and y limits to 0 and 1
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.axis('off')
        plt.savefig(f"../gif_aug/{pic[:-4]}_{frame-1}.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    col_pal = sns.color_palette("Spectral", n_colors=20)
    fig, ax = plt.subplots(ncols=1, figsize=(10, 8), dpi=300)
    xs = []
    ys = []
    for x in data[0][0::2]:
        xs.append(x)
    for y in data[0][1::2]:
        ys.append(1 - y)
    sns.scatterplot(x="x", y="y", palette=col_pal[0], data=pd.DataFrame({"x": xs, "y": ys}), ax=ax, legend=False)
    set_color(fig, ax)
    # set x and y limits to 0 and 1
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.axis('off')
    plt.savefig(f"../gif_aug/{pic[:-4]}_19.png", bbox_inches='tight', pad_inches=0)


    # create gif from folder gif with images
    images = []
    for file in range(20):
        images.append(imageio.v2.imread(f"../gif_aug/{pic[:-4]}_{file}.png"))
    imageio.mimsave(f"../gif_aug/{pic[:-4]}.gif", images, duration=0.1)
    plt.close(fig)
"""
