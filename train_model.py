import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.utils import to_categorical

# === Parameters ===
DATA_PATH = 'keypoints'
SEQUENCE_LENGTH = 30

def pad_or_truncate(sequence, target_len=SEQUENCE_LENGTH):
    length = sequence.shape[0]
    if length > target_len:
        return sequence[:target_len]
    elif length < target_len:
        padding = np.zeros((target_len - length, sequence.shape[1]))
        return np.vstack((sequence, padding))
    return sequence

# === Labels and Mapping ===
LABELS = sorted([
    label for label in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, label))
])
label_map = {label: idx for idx, label in enumerate(LABELS)}

# === Load Sequences and Labels ===
sequences, labels = [], []

for label in LABELS:
    folder_path = os.path.join(DATA_PATH, label)
    for file in os.listdir(folder_path):
        if file.endswith('.npy'):
            path = os.path.join(folder_path, file)
            sequence = np.load(path)

            if sequence.shape[1] != 126:
                print(f"⚠️ Skipped file due to feature mismatch: {sequence.shape} — {path}")
                continue

            # Optional: Track one-hand entries
            if np.count_nonzero(sequence[0][63:]) == 0:
                print(f"🖐️ One-hand sequence detected: {file}")

            padded_sequence = pad_or_truncate(sequence)
            sequences.append(padded_sequence)
            labels.append(label_map[label])

# === Convert to arrays ===
X = np.array(sequences)
X = X.reshape(len(X), SEQUENCE_LENGTH, 126)
y = to_categorical(labels).astype(int)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# === Build Model ===
model = Sequential()
model.add(Input(shape=(SEQUENCE_LENGTH, 126)))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(LABELS), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(f"✅ Loaded sequences: {len(sequences)}")
print(f"✅ Loaded labels:    {len(labels)}")
print(f"🧠 Classes:          {LABELS}")

# === Train Model ===
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# === Save Model ===
model.save('ngt_model.h5')
print("✅ Model saved as ngt_model.h5")

