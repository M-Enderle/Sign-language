import os.path

from utils import *


def augmentation(path=DATA_PATH, amount: int = 10):
    def boundaries(data):
        _x_min, _y_min, _x_max, _y_max = 1, 1, 0, 0
        for _frame in data:
            xs = _frame[0::2]
            ys = _frame[1::2]
            _x_min = min(_x_min, np.min(xs)) if np.min(xs) != 0.0 else _x_min
            _y_min = min(_y_min, np.min(ys)) if np.min(ys) != 0.0 else _y_min
            _x_max = max(_x_max, np.max(xs))
            _y_max = max(_y_max, np.max(ys))
        _x_min = _x_min if _x_min != 1 else 0.0
        _y_min = _y_min if _y_min != 1 else 0.0
        _x_max = _x_max if _x_max <= 1 else 1.0
        _y_max = _y_max if _y_max <= 1 else 1.0
        return _x_min, _y_min, _x_max, _y_max

    l_map = label_map()
    sequences, labels = [], []
    actions = get_actions()
    for action in actions:
        n_seq = len(os.listdir(os.path.join(path, action)))
        print("applying augmentation on folder", action)
        for seq in range(n_seq):
            move_dirs = random.choices(
                ["left", "right", "up", "down", "left-up", "right-up", "left-down", "right-down"], k=amount)
            for index, d in enumerate(move_dirs):
                data = np.load(os.path.join(path, action, f"{seq}.npy"))
                x_min, y_min, x_max, y_max = boundaries(data)
                to_move = random.random()
                s = []
                for frame in data:
                    if d == "left":
                        frame[0::2] -= to_move * x_min
                    elif d == "right":
                        frame[0::2] += to_move * (1 - x_max)
                    elif d == "up":
                        frame[1::2] += to_move * (1 - y_max)
                    elif d == "down":
                        frame[1::2] -= to_move * y_min
                    elif d == "left-up":
                        frame[0::2] -= to_move * x_min
                        frame[1::2] += to_move * (1 - y_max)
                    elif d == "right-up":
                        frame[0::2] += to_move * (1 - x_max)
                        frame[1::2] += to_move * (1 - y_max)
                    elif d == "left-down":
                        frame[0::2] -= to_move * x_min
                        frame[1::2] -= to_move * y_min
                    elif d == "right-down":
                        frame[0::2] += to_move * (1 - x_max)
                        frame[1::2] -= to_move * y_min
                    s.append(frame)
                if not os.path.exists(os.path.join(DATA_PATH, action)):
                    os.makedirs(os.path.join(DATA_PATH, action))
                sequences.append(s)
                labels.append(l_map[action])
    return sequences, labels


def augmentation_extreme_on_image(image=os.path.join(DATA_PATH, "world", "2.npy")):
    def boundaries(data):
        _x_min, _y_min, _x_max, _y_max = 1, 1, 0, 0
        for _frame in data:
            xs = _frame[0::2]
            ys = _frame[1::2]
            _x_min = min(_x_min, np.min(xs)) if np.min(xs) != 0.0 else _x_min
            _y_min = min(_y_min, np.min(ys)) if np.min(ys) != 0.0 else _y_min
            _x_max = max(_x_max, np.max(xs))
            _y_max = max(_y_max, np.max(ys))
        _x_min = _x_min if _x_min != 1 else 0.0
        _y_min = _y_min if _y_min != 1 else 0.0
        _x_max = _x_max if _x_max <= 1 else 1.0
        _y_max = _y_max if _y_max <= 1 else 1.0
        return _x_min, _y_min, _x_max, _y_max

    move_dirs = ["left", "right", "up", "down", "left-up", "right-up", "left-down", "right-down"]
    for index, d in enumerate(move_dirs):
        data = np.load(image)
        x_min, y_min, x_max, y_max = boundaries(data)
        s = []
        for frame in data:
            if d == "left":
                frame[0::2] -= x_min
            elif d == "right":
                frame[0::2] += (1 - x_max)
            elif d == "up":
                frame[1::2] += (1 - y_max)
            elif d == "down":
                frame[1::2] -= y_min
            elif d == "left-up":
                frame[0::2] -= x_min
                frame[1::2] += (1 - y_max)
            elif d == "right-up":
                frame[0::2] += (1 - x_max)
                frame[1::2] += (1 - y_max)
            elif d == "left-down":
                frame[0::2] -= x_min
                frame[1::2] -= y_min
            elif d == "right-down":
                frame[0::2] += (1 - x_max)
                frame[1::2] -= y_min
            s.append(frame)
        np.save(os.path.join(DATA_PATH, "world_aug", f"2_aug_{d}.npy"), s)
