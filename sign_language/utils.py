import os
import random
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
sequence_length = 20
blink_counter = 0
DATA_PATH = os.path.join('../data/')


def detect_landmarks(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results, face=True, pose=False, left_hand=True, right_hand=True):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              ) if face else None

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              ) if pose else None

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              ) if left_hand else None

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              ) if right_hand else None


def _generate_face(results):
    if not results.face_landmarks:
        return np.zeros(160)

    face = np.array([[res.x, res.y] for res in results.face_landmarks.landmark]) \
        if results.face_landmarks else np.zeros(468 * 3)
    x_min, x_max, y_min, y_max = np.min(face[:, 0]), np.max(face[:, 0]), \
                                 np.min(face[:, 1]), np.max(face[:, 1])
    closest_points = []

    for y in np.linspace(y_min, y_max, 20):
        closest_points.append(np.argmin(np.abs(face[:, 0] - x_min) + np.abs(face[:, 1] - y)))
        closest_points.append(np.argmin(np.abs(face[:, 0] - x_max) + np.abs(face[:, 1] - y)))

    for x in np.linspace(x_min, x_max, 20):
        closest_points.append(np.argmin(np.abs(face[:, 0] - x) + np.abs(face[:, 1] - y_max)))
        closest_points.append(np.argmin(np.abs(face[:, 0] - x) + np.abs(face[:, 1] - y_min)))

    return np.array([[face[i, 0], face[i, 1]] for i in closest_points]).flatten()


def create_numpy(results):
    lh = np.array([[res.x, res.y] for res in
                   results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 2)

    rh = np.array([[res.x, res.y] for res in
                   results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 2)

    return np.concatenate([_generate_face(results), lh, rh])


def visualize_probabilities(res, image, actions):
    i = 0
    for num, prob in enumerate(res):
        if not actions[num] == 'none':
            cv2.rectangle(image, (0, 60 + i * 20), (int(prob * 100), 80 + i * 20),
                          (0, prob * 150, (1 - prob) * 150), -1)
            cv2.putText(image, actions[num], (5, 75 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            i += 1


def label_map(path=DATA_PATH):
    return {label: num for num, label in enumerate(get_actions(path))}


def get_actions(path=DATA_PATH):
    actions = []
    files = os.listdir(path)
    files.sort()
    for file in files:
        if os.path.isdir(os.path.join(path, file)):
            actions.append(file)

    return np.array(actions)


def create_folders(actions, path=DATA_PATH):
    if not os.path.exists(path):
        os.makedirs(path)

    for action in actions:
        if not os.path.exists(os.path.join(path, action)):
            os.makedirs(os.path.join(path, action))


def load_numpy(path=DATA_PATH):
    sequences, labels = [], []
    for action in get_actions(path):
        for file in os.listdir(os.path.join(path, action)):
            window = list(np.load(os.path.join(path, action, file)))
            sequences.append(window)
            labels.append(label_map()[action])
    return sequences, labels


def show_sentence(image, sentence):
    global blink_counter

    roi = image[0:70, 0:image.shape[1]]
    roi = cv2.GaussianBlur(roi, (71, 71), 0)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGBA)
    roi = cv2.addWeighted(roi, 0.5, np.zeros(roi.shape, dtype=np.uint8), 0, 0)
    roi = cv2.cvtColor(roi, cv2.COLOR_RGBA2BGR)
    image[0:70, 0:image.shape[1]] = roi

    # write the sentence to the video
    if sentence:
        text = " ".join(sentence) + " "
    else:
        text = ""

    if blink_counter % 17 <= 8:
        text += "_"

    blink_counter += 1

    cv2.putText(image, text, (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 1, cv2.LINE_AA)


def remove_unnec(to_rem: list):
    data_from_to = {
        "face": (0, 160),
        "lh": (160, 202),
        "rh": (202, 244)
    }
    to_rem.sort()
    actions = get_actions()
    for action in actions:
        n_seq = len(os.listdir(os.path.join(DATA_PATH, action)))
        for seq in range(n_seq):
            s = []
            data = np.load(os.path.join(DATA_PATH, action, f"{seq}.npy"))
            for frame in data:
                am_rem = 0
                for r in to_rem:
                    if r in data_from_to:
                        frame = np.delete(frame, [range(data_from_to[r][0] - am_rem, data_from_to[r][1] - am_rem)])
                        am_rem += data_from_to[r][1] - data_from_to[r][0] + 1
                s.append(frame)
            npy_path = os.path.join(DATA_PATH, action, str(seq))
            np.save(npy_path, np.array(s))
