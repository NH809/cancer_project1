import os
import cv2
import numpy as np

# Dataset path
data_dir = "dataset"

# Categories (folder names)
categories = ["glioma", "meningioma", "pituitary", "notumor"]

# Image size
IMG_SIZE = 224

data = []
labels = []

# Loop through folders
for category in categories:
    path = os.path.join(data_dir, category)
    class_num = categories.index(category)

    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path)

            # Resize image
            resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

            data.append(resized)
            labels.append(class_num)

        except Exception as e:
            pass

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Normalize data (0 to 1)
data = data / 255.0

print("✅ Total images loaded:", len(data))
print("✅ Labels created:", len(labels))
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, 4)
y_test = to_categorical(y_test, 4)

print("✅ Data split completed")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("✅ Model created successfully")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save("model.keras")
print("✅ Model saved successfully")