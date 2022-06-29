import os.path

import numpy as np

from utils import *


def augmentation_dir(path=DATA_PATH, amount: int = 10):
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


def augmentation(path=DATA_PATH, amount: int = 10):
    def get_factor(_data):
        _x_min, _y_min, _x_max, _y_max = 1, 1, 0, 0
        for _frame in _data:
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

        center = np.array([(_x_min + _x_max) / 2, (_y_min + _y_max) / 2])

        max_factor = np.min([center[0] / (center[0] - _x_min),
                             center[1] / (center[1] - _y_min),
                             (1 - center[0]) / (_x_max - center[0]),
                             (1 - center[1]) / (_y_max - center[1])])
        factor = random.uniform(0.5, max_factor)

        return center, factor

    def boundaries(_frame):
        _x_min, _y_min, _x_max, _y_max = 1, 1, 0, 0
        xs = _frame[0::2]
        ys = _frame[1::2]
        _x_min = min(_x_min, np.min(xs)) if np.min(xs) != 0.0 else _x_min if _x_min != 1 else 0.0
        _y_min = min(_y_min, np.min(ys)) if np.min(ys) != 0.0 else _y_min if _y_min != 1 else 0.0
        _x_max = max(_x_max, np.max(xs)) if _x_max <= 1 else 1.0
        _y_max = max(_y_max, np.max(ys)) if _y_max <= 1 else 1.0
        return _x_min, _y_min, _x_max, _y_max

    def move_frame(_frame, _dir, x_to_move, y_to_move):
        _x_min, _y_min, _x_max, _y_max = boundaries(_frame)

        if _dir == "left":
            frame[0::2] -= x_to_move * _x_min
        elif _dir == "right":
            frame[0::2] += x_to_move * (1 - _x_max)
        elif _dir == "up":
            frame[1::2] += y_to_move * (1 - _y_max)
        elif _dir == "down":
            frame[1::2] -= y_to_move * _y_min
        elif _dir == "left-up":
            frame[0::2] -= x_to_move * _x_min
            frame[1::2] += y_to_move * (1 - _y_max)
        elif _dir == "right-up":
            frame[0::2] += x_to_move * (1 - _x_max)
            frame[1::2] += y_to_move * (1 - _y_max)
        elif _dir == "left-down":
            frame[0::2] -= x_to_move * _x_min
            frame[1::2] -= y_to_move * _y_min
        elif _dir == "right-down":
            frame[0::2] += x_to_move * (1 - _x_max)
            frame[1::2] -= y_to_move * _y_min
        return frame

    l_map = label_map()
    sequences, labels = [], []
    actions = get_actions()
    for action in actions:
        n_seq = len(os.listdir(os.path.join(path, action)))
        print("applying augmentation on folder", action)
        for seq in range(n_seq):
            for _ in range(amount):
                data = np.load(os.path.join(path, action, f"{seq}.npy"))
                center, factor = get_factor(data)
                s = []
                x_to_move = random.random()
                y_to_move = random.random()
                dir = random.choice(["left", "right", "up", "down", "left-up", "right-up", "left-down", "right-down"])
                for frame in data:
                    # assert max(frame[0::2]) <= 1.0
                    # assert max(frame[1::2]) <= 1.0
                    # assert min(frame[0::2]) >= 0.0
                    # assert min(frame[1::2]) >= 0.0

                    frame[0::2] -= center[0]
                    frame[1::2] -= center[1]

                    frame[0::2] *= factor
                    frame[1::2] *= factor

                    frame[0::2] += center[0]
                    frame[1::2] += center[1]

                    # assert max(frame[0::2]) <= 1.0
                    # assert max(frame[1::2]) <= 1.0
                    # assert min(frame[0::2]) >= 0.0
                    # assert min(frame[1::2]) >= 0.0

                    frame = move_frame(frame, dir, x_to_move, y_to_move)

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
        return _x_min, _y_min, _x_max, _y_max  #

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


def augmentation_size_on_image(image=os.path.join(DATA_PATH, "world", "2.npy"), amount: int = 10):
    def get_factor(_data):
        _x_min, _y_min, _x_max, _y_max = 1, 1, 0, 0
        for _frame in _data:
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

        center = np.array([(_x_min + _x_max) / 2, (_y_min + _y_max) / 2])

        max_factor = np.min([center[0] / (center[0] - _x_min),
                             center[1] / (center[1] - _y_min),
                             (1 - center[0]) / (_x_max - center[0]),
                             (1 - center[1]) / (_y_max - center[1])])
        factor = random.uniform(0, max_factor)

        return center, factor

    l_map = label_map()
    sequences, labels = [], []

    for i in range(amount):
        data = np.load(image)
        center, factor = get_factor(data)
        s = []
        for frame in data:
            frame_ = frame.copy()
            frame[0::2] -= center[0]
            frame[1::2] -= center[1]

            frame[0::2] *= factor
            frame[1::2] *= factor

            frame[0::2] += center[0]
            frame[1::2] += center[1]

            assert max(frame[0::2]) <= 1.0
            assert max(frame[1::2]) <= 1.0
            assert min(frame[0::2]) >= 0.0
            assert min(frame[1::2]) >= 0.0

            assert not np.equal(frame, frame_).all()

            s.append(frame)
        np.save(os.path.join("../size_aug", f"2_size_{i}_aug.npy"), s)
