import numpy as np
from tensorflow.keras.preprocessing import image

def predict_image(path, model):
    img = image.load_img(path, target_size=(224,224))
    arr = image.img_to_array(img)/255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)

    classes = ['glioma','meningioma','notumor','pituitary']
    idx = np.argmax(pred)

    confidence = float(pred[0][idx]) * 100
    label = classes[idx]

    if label == "notumor":
        result = "No Cancer"
        stage = "Healthy"
    else:
        result = f"Tumor Detected ({label})"

        if confidence < 70:
            stage = "Low Risk"
        elif confidence < 85:
            stage = "Moderate Risk"
        else:
            stage = "High Risk"

    return label, confidence, arr, result, stage