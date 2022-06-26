from datetime import datetime, timedelta
from keras.models import load_model
from utils import *

model = load_model('../data/model.h5')
actions = get_actions()

threshold = 0.97

last_change = datetime.now()
sentence = []
sequence = []


cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image, results = detect_landmarks(frame, holistic)
        draw_landmarks(image, results)

        key_points = create_numpy(results)

        sequence.append(key_points)
        sequence = sequence[-sequence_length:]

        if len(sequence) == sequence_length:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            if np.max(res) > threshold:
                if actions[np.argmax(res)] == "none":
                    pass
                elif sentence and sentence[-1] != actions[np.argmax(res)]:
                    sentence.append(actions[np.argmax(res)])
                    last_change = datetime.now()
                elif not sentence:
                    sentence.append(actions[np.argmax(res)])
                    last_change = datetime.now()

            visualize_probabilities(res, image, actions)

        if datetime.now() - last_change > timedelta(seconds=5):
            sentence = []
            last_change = datetime.now()

        image = cv2.resize(image, (0,0), fx=1.3, fy=1.3)

        show_sentence(image, sentence)
        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
