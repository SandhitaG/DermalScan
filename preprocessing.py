import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ------------------------------------------
# PATHS & SETTINGS
# ------------------------------------------
DATASET_DIR = "dataset"     # main folder containing class folders
IMG_SIZE = 224              # EfficientNetB0 input size

haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# ------------------------------------------
# READ + FACE DETECT + CROP + RESIZE
# ------------------------------------------
images = []
labels = []
class_names = os.listdir(DATASET_DIR)

for label_id, class_name in enumerate(class_names):
    class_folder = os.path.join(DATASET_DIR, class_name)
    
    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)

        # read image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect face
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            img = img[y:y+h, x:x+w]   # crop face

        # resize to 224x224
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # normalize (0-1)
        img = img.astype("float32") / 255.0

        images.append(img)
        labels.append(label_id)

print("Total images loaded:", len(images))

# ------------------------------------------
# LABEL ENCODING (One-hot)
# ------------------------------------------
labels = to_categorical(labels, num_classes=len(class_names))

# ------------------------------------------
# TRAIN-TEST SPLIT
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    np.array(images),
    np.array(labels),
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ------------------------------------------
# IMAGE AUGMENTATION
# ------------------------------------------
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# final training generator
train_generator = aug.flow(X_train, y_train, batch_size=8)

print("Preprocessing Completed Successfully!")
