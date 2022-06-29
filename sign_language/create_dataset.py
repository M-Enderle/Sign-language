import cv2

from utils import *

actions = np.array(['project goals'])
no_sequences = 10
sequence_length = 20

create_folders(actions, DATA_PATH)

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
    for action in actions:
        while not cv2.waitKey(10) & 0xFF == ord('c'):

            ret, frame = cap.read()

            cv2.putText(frame, '{}, press c to start'.format(action), (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', frame)
        cv2.waitKey(1000)
        for sequence in range(no_sequences):

            numpy_seq = []
            """
            if sequence != 0 and sequence % 40 == 0:
                while not cv2.waitKey(10) & 0xFF == ord('c'):
                    ret, frame = cap.read()
                    cv2.putText(image, 'Take a break. press c to continue', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)
            """
            while True:

                if len(numpy_seq) >= sequence_length:
                    break

                ret, frame = cap.read()
                image, results = detect_landmarks(frame, holistic)
                draw_landmarks(image, results)
                if len(numpy_seq) == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(1000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(10)

                if results.right_hand_landmarks or results.left_hand_landmarks:
                    key_points = create_numpy(results)
                    numpy_seq.append(key_points)

            npy_path = os.path.join(DATA_PATH, action, str(sequence))
            np.save(npy_path, np.array(numpy_seq))

    cap.release()
    cv2.destroyAllWindows()
