import os.path

from utils import *


def augmentation(path=DATA_PATH, amount: int = 10):
    def get_mins_and_maxs(data):
        _x_min, _y_min, _x_max, _y_max = 1, 1, 0, 0
        for frame in data:
            xs = frame[0::2]
            ys = frame[1::2]
            _x_min = min(_x_min, np.min(xs)) if np.min(xs) != 0.0 else _x_min
            _y_min = min(_y_min, np.min(ys)) if np.min(ys) != 0.0 else _y_min
            _x_max = max(_x_max, np.max(xs))
            _y_max = max(_y_max, np.max(ys))
        _x_min = _x_min if _x_min != 1 else 0.0
        _y_min = _y_min if _y_min != 1 else 0.0
        _x_max = _x_max if _x_max <= 1 else 1.0
        _y_max = _y_max if _y_max <= 1 else 1.0
        return _x_min, _y_min, _x_max, _y_max

    sequences, labels = [], []
    actions = get_actions()
    for action in actions:
        n_seq = len(os.listdir(os.path.join(path, action)))
        print(action)
        for seq in range(n_seq):
            move_dirs = random.choices(["left", "right", "up", "down"], k=amount)
            data = np.load(os.path.join(path, action, f"{seq}.npy"))
            x_min, y_min, x_max, y_max = get_mins_and_maxs(data)
            for index, d in enumerate(move_dirs):
                s = []
                for frame in data:
                    if d == "left":
                        frame[0::2] -= random.random() * x_min
                    elif d == "right":
                        frame[0::2] += random.random() * (1 - x_max)
                    elif d == "up":
                        frame[1::2] += random.random() * (1 - y_max)
                    elif d == "down":
                        frame[1::2] -= random.random() * y_min
                    s.append(frame)
                if not os.path.exists(os.path.join(DATA_PATH, action)):
                    os.makedirs(os.path.join(DATA_PATH, action))
                sequences.append(s)
                labels.append(action)
    return sequences, labels
