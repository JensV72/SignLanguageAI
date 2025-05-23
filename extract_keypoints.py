import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# === Paths ===
DATASET_DIR = './dataset'
OUTPUT_DIR = './keypoints'

# === MediaPipe Hands Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        lh = np.zeros(21 * 3)
        rh = np.zeros(21 * 3)

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                
                if handedness == 'Left':
                    lh = keypoints
                elif handedness == 'Right':
                    rh = keypoints

        frame_keypoints = np.concatenate([lh, rh])
        keypoints_sequence.append(frame_keypoints)

    cap.release()
    return np.array(keypoints_sequence)


# === Run Extraction ===
for label in os.listdir(DATASET_DIR):
    label_dir = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    output_label_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(output_label_dir, exist_ok=True)

    for filename in tqdm(os.listdir(label_dir), desc=f"Processing {label}"):
        if filename.endswith('.mp4'):
            video_path = os.path.join(label_dir, filename)
            keypoints = extract_keypoints_from_video(video_path)

            save_path = os.path.join(output_label_dir, filename.replace('.mp4', '.npy'))
            np.save(save_path, keypoints)

