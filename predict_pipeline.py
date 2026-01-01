import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
MODEL_PATH = "/content/dermalscan_efficientnetb0.h5"  # change if needed
AGE_MODEL_PATH = None  # optional
IMG_SIZE = 224
CLASS_NAMES = ['Wrinkles', 'clear skin', 'dark spots', 'puffy eyes']
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
PADDING = 0.25  # expand face bbox by 25%

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------
def load_models(model_path=MODEL_PATH, age_model_path=AGE_MODEL_PATH):
    print("Loading classification model...")
    clf = load_model(model_path)

    age_model = None
    if age_model_path and os.path.exists(age_model_path):
        print("Loading age model...")
        age_model = load_model(age_model_path)
    else:
        print("No age model provided.")

    return clf, age_model

# --------------------------------------------------
# FACE BBOX EXPANSION
# --------------------------------------------------
def expand_bbox(x, y, w, h, img_w, img_h):
    pad_w = int(w * PADDING)
    pad_h = int(h * PADDING)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    return x1, y1, x2, y2

# --------------------------------------------------
# PREPROCESS PATCH FOR MODEL
# --------------------------------------------------
def preprocess_patch(patch):
    if patch is None:
        return None
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch = cv2.resize(patch, (IMG_SIZE, IMG_SIZE))
    arr = np.asarray(patch).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

# --------------------------------------------------
# MAKE A PREDICTION ON ONE REGION
# --------------------------------------------------
def predict_patch(model, patch):
    arr = preprocess_patch(patch)
    if arr is None:
        return None, None

    probs = model.predict(arr)[0]
    top_idx = np.argsort(probs)[::-1]  # descending order
    return probs, top_idx

# --------------------------------------------------
# MAIN DETECTION + PREDICTION FUNCTION
# --------------------------------------------------
def detect_and_predict(image_path, clf_model, age_model=None,
                       out_path="annotated_output.jpg", save_output=True):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Cannot open image: " + image_path)

    original = img.copy()
    img_h, img_w = img.shape[:2]

    face_cascade = cv2.CascadeClassifier(HAAR_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Fallback if no face found
    regions = []
    if len(faces) == 0:
        print("No face detected. Using full image.")
        regions.append((0, 0, img_w, img_h))
    else:
        print(f"{len(faces)} face(s) detected.")
        for (x, y, w, h) in faces:
            x1, y1, x2, y2 = expand_bbox(x, y, w, h, img_w, img_h)
            regions.append((x1, y1, x2 - x1, y2 - y1))

    # Predict for each detected region
    for (x, y, w, h) in regions:
        patch = original[y:y+h, x:x+w]

        probs, top_idx = predict_patch(clf_model, patch)
        if probs is None:
            continue

        # Format predictions
        top3 = top_idx[:3]
        labels_text = []
        for idx in top3:
            label = CLASS_NAMES[idx]
            pct = probs[idx] * 100
            labels_text.append(f"{label}: {pct:.1f}%")

        # Optional age prediction
        age_text = "Age: N/A"
        if age_model is not None:
            age_arr = preprocess_patch(patch)
            age_pred = age_model.predict(age_arr)[0]
            age_text = f"Age: {int(age_pred)}"

        # Draw bounding box + predictions
        cv2.rectangle(original, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(original, labels_text[0], (x, y-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        ypos = y - 10
        for text in labels_text[1:]:
            cv2.putText(original, text, (x, ypos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1)
            ypos += 18

        cv2.putText(original, age_text, (x, y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)

    # Save output
    if save_output:
        cv2.imwrite(out_path, original)
        print("Annotated image saved to:", out_path)

    return original
# --------------------------------------------------
# END OF FILE
# --------------------------------------------------
