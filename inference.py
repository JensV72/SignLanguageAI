import cv2
import mediapipe as mp
import numpy as np
from pythonosc.udp_client import SimpleUDPClient
from tensorflow.keras.models import load_model

# OSC Client setup (sends to TouchDesigner)
client = SimpleUDPClient("127.0.0.1", 8000)

# Load trained LSTM model
model = load_model('ngt_model.h5')

# Labels used during training (must match model)
LABELS = ['huilen', 'knuffel', 'verdrietig', 'verliefd']

# Parameters
SEQUENCE_LENGTH = 30
BUFFER = []

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start webcam
cap = cv2.VideoCapture(0)

def pad_or_truncate(sequence, target_len=SEQUENCE_LENGTH):
    length = sequence.shape[0]
    if length > target_len:
        return sequence[:target_len]
    elif length < target_len:
        padding = np.zeros((target_len - length, sequence.shape[1]))
        return np.vstack((sequence, padding))
    return sequence

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hands_detected = results.multi_hand_landmarks
        data_aux = []

        for hand_landmarks in hands_detected[:2]:
            x_, y_ = [], []
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.extend([
                    lm.x - min(x_),
                    lm.y - min(y_),
                    lm.z
                ])

        # Pad with zeros if only 1 hand (63 features)
        if len(data_aux) == 63:
            data_aux.extend([0.0] * 63)

        # Only add to buffer if data is complete
        if len(data_aux) == 126:


            BUFFER.append(data_aux)
            print(f"🟡 BUFFER size: {len(BUFFER)}")

            if len(BUFFER) == SEQUENCE_LENGTH:
                input_seq = np.array(BUFFER)
                input_seq = pad_or_truncate(input_seq)
                input_seq = input_seq.reshape(1, SEQUENCE_LENGTH, 126)

                # Inspect input shape
                print(f" input_seq shape: {input_seq.shape}")  # Expect (1, 30, 126)

                # Predict and inspect probabilities
                prediction = model.predict(input_seq)[0]  # Get the first batch result

                # Print full prediction confidence per class
                print("Prediction confidence:")
                for i, label in enumerate(LABELS):
                    print(f"   {label}: {prediction[i]:.2f}")

                # Send only if confidence is high enough
                predicted_index = np.argmax(prediction)
                confidence = prediction[predicted_index]
                predicted_label = LABELS[predicted_index]

                if confidence > 0.8:
                    print(f"High confidence ({confidence:.2f}) — sending: {predicted_label}")
                    client.send_message("/sign", predicted_label)
                else:
                    print(f"Low confidence ({confidence:.2f}) — not sending")


                # Display label on frame
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                cv2.putText(frame, predicted_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                BUFFER = []

    # Draw landmarks on hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    # Show the frame
    cv2.imshow("Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

