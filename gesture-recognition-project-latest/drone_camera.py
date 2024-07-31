import csv
import copy
import argparse
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp
from utils import CvFpsCalc
from collections import deque
from collections import Counter
from model import KeyPointClassifier
from model import PointHistoryClassifier
from djitellopy import Tello

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", help='min_detection_confidence', type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", help='min_tracking_confidence', type=float, default=0.5)
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    use_brect = True

    # Initialize Tello drone
    tello = Tello()
    tello.connect()
    tello.streamon()

    # Camera preparation
    cap = tello.get_frame_read()

    # Model load
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Read the labels from the csv files (keypoint and point history files).
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]

    # Measure frame rate of camera.
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history
    finger_gesture_history = deque(maxlen=history_length)

    mode = -1  # Set the initial mode to 0 (default mode)

    while True:
        fps = cvFpsCalc.get()

        # "ESC" key -> ends program
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Using the drone's camera to capture data.
        frame = cap.frame
        if frame is None:
            break
        image = cv.flip(frame, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                try:
                    # Calculating the bounding box.
                    brect = calc_box(debug_image, hand_landmarks)

                    # Calculating the landmarks.
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                    # Write to the dataset file
                    log_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == "Not applicable":  # Point gesture
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])

                    # Finger gesture classification
                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                    # Calculates the gesture IDs in the latest detection
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(finger_gesture_history).most_common()

                    # Execute gesture control
                    hand_sign_text = keypoint_classifier_labels[hand_sign_id]
                    gesture_control(tello, hand_sign_text)

                    # Drawing part
                    debug_image = draw_box(use_brect, debug_image, brect)
                    debug_image = create_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        hand_sign_text,
                        point_history_classifier_labels[most_common_fg_id[0][0]],
                    )
                except Exception as e:
                    print(f"Error processing hand landmarks: {e}")

        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection
        cv.imshow('Hand Gesture Recognition', debug_image)

    tello.streamoff()
    tello.end()

def gesture_control(tello, hand_sign_text):
    try:
        if hand_sign_text == 'Take Off':
            tello.takeoff()
            tello.hover()
        elif hand_sign_text == 'Land':
            tello.land()
        elif hand_sign_text == 'Up':
            tello.move_up(30)
        elif hand_sign_text == 'Down':
            tello.move_down(30)
        elif hand_sign_text == 'Left':
            tello.move_left(30)
        elif hand_sign_text == 'Right':
            tello.move_right(50)
        elif hand_sign_text == 'Forward':
            tello.move_forward(50)
        elif hand_sign_text == 'Back':
            tello.move_back(50)
        # elif hand_sign_text == 'Victory':
        #     tello.rotate_clockwise(30)
        # elif hand_sign_text == 'Stop':
        #     tello.hover()
    except Exception as e:
        print(f"Error in gesture control: {e}")

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 - 9
        number = key - 48
    if key == 110:  # 'n' key: resets mode back to normal
        mode = 0
    if key == 107:  # 'k' key: logging key points
        mode = 1
    if key == 104:  # 'h' key: point history
        mode = 2
    return number, mode

def calc_box(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    # Gets the absolute distance (magnitude) and normalizes the coordinates
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    normalized_landmark_list = list(map(normalize_, temp_landmark_list))

    return normalized_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

def log_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *pre_processed_landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history_classifier_label.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *pre_processed_point_history_list])

def draw_box(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)

    return image

def create_landmarks(image, landmark_list):
    if len(landmark_list) > 0:
        for landmark in landmark_list:
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 255, 0), -1)

    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    info_text = handedness.classification[0].label[0:]
    info_text += ':' + hand_sign_text

    cv.putText(image, info_text, (brect[0], brect[1] - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv.LINE_AA)

    return image

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + index, (152, 251, 152), 2)

    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 0 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode], (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "MODE:" + mode_string[mode], (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

    if 0 <= number <= 9:
        cv.putText(image, "NUM:" + str(number), (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "NUM:" + str(number), (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

    return image

if __name__ == '__main__':
    main()
