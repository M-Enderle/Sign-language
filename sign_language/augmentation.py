import os.path
from utils import *


def get_borders(data):
    x_min, y_min, x_max, y_max = 1, 1, 0, 0

    for frame in data:
        x = frame[0::2]
        y = frame[1::2]
        x_min = min(x_min, np.min(x)) if np.min(x) != 0.0 else x_min
        y_min = min(y_min, np.min(y)) if np.min(y) != 0.0 else y_min
        x_max = max(x_max, np.max(x))
        y_max = max(y_max, np.max(y))

    x_min = x_min if x_min != 1 else 0.0
    y_min = y_min if y_min != 1 else 0.0
    x_max = x_max if x_max <= 1 else 1.0
    y_max = y_max if y_max <= 1 else 1.0

    return x_min, y_min, x_max, y_max


def scale(data):

    x_min, y_min, x_max, y_max = get_borders(data)

    center = [(x_min + x_max) / 2, (y_min + y_max) / 2]

    if center[0] == 0.0 or center[1] == 0.0:
        return data

    max_factor = np.min([center[0] / (center[0] - x_min),
                         center[1] / (center[1] - y_min),
                         (1 - center[0]) / (x_max - center[0]),
                         (1 - center[1]) / (y_max - center[1])])

    factor = random.uniform(0.5, max_factor)

    for frame in data:
        x = frame[0::2]
        y = frame[1::2]
        
        x[x != 0] -= center[0]
        y[y != 0] -= center[1]

        x[x != 0] *= factor
        y[y != 0] *= factor

        x[x != 0] += center[0]
        y[y != 0] += center[1]

        x[x != 0] = np.maximum(x[x != 0], 0)
        y[y != 0] = np.maximum(y[y != 0], 0)

    return data


def move(data):
    x_min, y_min, x_max, y_max = get_borders(data)
    random_movement_x = (random.random() * (x_min + (1 - x_max))) - x_min
    random_movement_y = (random.random() * (y_min + (1 - y_max))) - y_min
    for frame in data:
        x = frame[0::2]
        y = frame[1::2]
        x[x != 0] += random_movement_x
        y[y != 0] += random_movement_y

    return data


def plot(data, sub_folder):
    from matplotlib import pyplot as plt
    import seaborn as sns
    import pandas as pd

    fig, ax = plt.subplots(ncols=1, figsize=(10, 8), dpi=75)
    col_pal = sns.color_palette("Spectral", n_colors=len(data))
    for index, frame in enumerate(data):
        x = []
        y = []
        colors = []
        for _x in frame[0::2]:
            x.append(_x)
            colors.append(col_pal[index])
        for _y in frame[1::2]:
            y.append(1 - _y)

        sns.scatterplot(x="x", y="y", palette="colors",
                        data=pd.DataFrame({"x": x, "y": y, "colors": colors}),
                        ax=ax, legend=False)

    set_color(fig, ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.axis('off')

    if not os.path.exists(f"../images/{sub_folder}"):
        os.makedirs(f"../images/{sub_folder}")

    files = os.listdir(f"../images/{sub_folder}")
    highest = max(int(file.split(".")[0]) for file in files) if files else 0
    plt.savefig(f"../images/{sub_folder}/{highest + 1}.png")
    plt.close()


def apply(path=DATA_PATH, amount: int = 10, test=None):
    sequences, labels = [], []
    actions = get_actions()
    l_map = label_map(path)
    for action in actions:
        files = os.listdir(os.path.join(path, action))
        print("applying augmentation on folder", action)
        for seq, filename in enumerate(files):
            for i in range(amount):
                data = np.load(os.path.join(path, action, filename))
                data = scale(data)
                data = move(data)
                sequences.append(data)
                labels.append(l_map[action])
                plot(data, seq) if test is not None else None

            if test is not None and seq == test:
                return None
    return sequences, labels


if __name__ == "__main__":
    for file in os.listdir("../images/"):
        for image in os.listdir(f"../images/{file}"):
            os.remove(f"../images/{file}/{image}")
    apply(test=4)