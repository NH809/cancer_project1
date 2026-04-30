import numpy as np
from tensorflow.keras.preprocessing import image

def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)

    return predicted_class, confidence

heatmap = generate_heatmap(img, model)

img_original = cv2.imread(path)
heatmap = cv2.resize(heatmap, (img_original.shape[1], img_original.shape[0]))
heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = cv2.addWeighted(img_original, 0.6, heatmap, 0.4, 0)

heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], "heat_" + file.filename)
cv2.imwrite(heatmap_path, superimposed_img)

print("RAW PREDICTION:", pred)

if not is_mri_image(path):
    os.remove(path)
    return render_template("index.html", error="❌ Invalid Image! Please upload MRI scan only.")