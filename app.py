from flask import Flask, render_template, request, redirect, session
import os, cv2
import numpy as np
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from reportlab.platypus import SimpleDocTemplate, Paragraph
import mysql.connector
import tensorflow as tf

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = os.path.join(os.getcwd(), "model.h5")

model = None
model_path = "model.h5"

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1uevBD7YUZy9U5nGlHC1Xe3_mCNY2iwju"
    gdown.download(url, model_path, quiet=False)

if os.path.exists(model_path):
    model = load_model(model_path, compile=False, safe_mode=False)
    print("✅ Model Loaded")
else:
    print("❌ Model not found")

# ================= LOAD MODEL =================


# ================= DATABASE =================
try:
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Nikita@2005",
        database="cancer_db"
    )
    cursor = db.cursor()
    print("✅ DB Connected")
except:
    db = None
    cursor = None
    print("⚠️ DB Not Connected (Running without DB)")

# ================= MRI VALIDATION =================
def is_mri_image(path):
    img = cv2.imread(path)

    if img is None:
        return False

    color_std = np.std(img)

    if color_std > 50:
        return False

    return True

# ================= AI DOCTOR NOTE =================
def generate_doctor_note(label, confidence, stage):

    if label == "notumor":
        return """
🟢 No tumor detected

🧠 Brain appears normal.

✔ Maintain healthy lifestyle  
✔ Routine checkups recommended  
"""

    return f"""
🔬 Tumor Type: {label.upper()}

📊 Confidence: {round(confidence,2)}%

🧠 Interpretation:
MRI scan shows abnormal tissue growth.

⚠️ Risk Level: {stage}

💊 Recommendation:
- Consult neurologist immediately
- MRI contrast scan advised
- Treatment: Surgery / Radiation / Chemotherapy

📍 Tumor Region:
Highlighted heatmap shows tumor activity
"""

# ================= LOGIN =================
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        cursor.execute(
            "SELECT id, username, password, role FROM users WHERE username=%s AND password=%s",
            (username, password)
        )
        user = cursor.fetchone()

        if user:
            session['user'] = user[1]
            session['role'] = user[3]

            if user[3] == "admin":
                return redirect('/admin')
            elif user[3] == "doctor":  # ✅ FIX HERE
                return redirect('/doctor')
            else:
                return redirect('/')
        else:
            return render_template("login.html", error="Invalid Login")

    return render_template("login.html")

# ================= LOGOUT =================
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# ================= HOME =================
@app.route('/')
def home():
    if 'user' not in session:
        return redirect('/login')
    return render_template("index.html")

# ================= ADMIN =================
@app.route('/admin')
def admin():
    if session.get('role') != 'admin':
        return redirect('/login')

    if cursor:
        cursor.execute("SELECT * FROM reports ORDER BY id DESC")
        data = cursor.fetchall()
    else:
        data = []

    return render_template("admin.html", data=data)

# ================= DOCTOR =================
@app.route('/doctor')
def doctor():
    if session.get('role') != 'doctor':
        return redirect('/login')

    if cursor:
        cursor.execute("SELECT * FROM reports ORDER BY id DESC")
        data = cursor.fetchall()
    else:
        data = []

    return render_template("doctor.html", data=data)

@app.route('/add_note/<int:id>', methods=['POST'])
def add_note(id):
    note = request.form.get('note')

    if cursor:
        cursor.execute(
            "UPDATE reports SET doctor_notes=%s WHERE id=%s",
            (note, id)
        )
        db.commit()

    return redirect('/doctor')

# ================= PREDICTION =================
def predict_image(path):
    img = image.load_img(path, target_size=(224,224))
    arr = image.img_to_array(img)/255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)
    print("Prediction:", pred)

    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

    label_index = np.argmax(pred)
    confidence = float(pred[0][label_index]) * 100
    label = classes[label_index]

    if confidence < 50:
        result = "Uncertain Result"
        stage = "Re-test Required"

    elif label == "notumor":
        result = "No Cancer"
        stage = "Healthy"

    else:
        result = f"Tumor Detected ({label})"

        if confidence < 70:
            stage = "Low Risk"
        elif confidence < 85:
            stage = "Moderate Risk"
        else:
            stage = "High Risk (Consult Doctor)"

    return label, confidence, arr, result, stage

# ================= GRADCAM =================
def generate_gradcam(model, img_array, last_conv_layer_name="conv2d_2"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

# ================= PREDICT =================
@app.route('/predict', methods=['POST'])
def predict():

    if 'user' not in session:
        return redirect('/login')

    name = request.form.get('name')
    file = request.files.get('file')

    if not file or file.filename == "":
        return render_template("index.html", error="Upload image")

    if not file.filename.lower().endswith(('.png','.jpg','.jpeg')):
        return render_template("index.html", error="Only image files allowed")

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    if not is_mri_image(path):
        os.remove(path)
        return render_template("index.html", error="❌ Invalid MRI Image")

    label, confidence, arr, result, stage = predict_image(path)

    # ✅ AI Doctor Note
    doctor_note = generate_doctor_note(label, confidence, stage)

    # ===== HEATMAP =====
    heat_name = None
    try:
        heatmap = generate_gradcam(model, arr)

        img_original = cv2.imread(path)
        heatmap = cv2.resize(heatmap, (img_original.shape[1], img_original.shape[0]))
        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(img_original, 0.6, heatmap, 0.4, 0)

        heat_name = "heat_" + file.filename
        cv2.imwrite(os.path.join(HEATMAP_FOLDER, heat_name), superimposed)

    except Exception as e:
        print("Heatmap Error:", e)

    if cursor:
        cursor.execute(
            "INSERT INTO reports (name,image,result,confidence,stage) VALUES (%s,%s,%s,%s,%s)",
            (name, file.filename, result, confidence, stage)
        )
        db.commit()

    return render_template("index.html",
        name=name,
        prediction=result,
        confidence=round(confidence,2),
        stage=stage,
        image=file.filename,
        heatmap=heat_name,
        doctor_note=doctor_note,
        chart_value=round(confidence,2),
        chart_remain=round(100-confidence,2)
    )

# ================= HISTORY =================
@app.route('/history')
def history():
    if 'user' not in session:
        return redirect('/login')

    if cursor:
        cursor.execute("SELECT * FROM reports ORDER BY id DESC")
        data = cursor.fetchall()
    else:
        data = []
    return render_template("history.html", data=data)

# ================= SEARCH =================
@app.route('/search', methods=['POST'])
def search():
    keyword = request.form.get('search')

    if cursor:
        cursor.execute(
            "SELECT * FROM reports WHERE name LIKE %s ORDER BY id DESC",
            ('%' + keyword + '%',)
        )
        data = cursor.fetchall()
    else:
        data = []
    return render_template("history.html", data=data)

# ================= DASHBOARD =================
@app.route('/dashboard')
def dashboard():
    if cursor:
        cursor.execute("SELECT result, COUNT(*) FROM reports GROUP BY result")
        data = cursor.fetchall()
    else:
        data = []

    labels = [i[0] for i in data]
    values = [i[1] for i in data]

    return render_template("dashboard.html", labels=labels, values=values)

# ================= PDF =================
@app.route('/download/<name>/<result>/<confidence>/<stage>')
def download(name, result, confidence, stage):

    filename = f"static/report_{name}.pdf"

    doc = SimpleDocTemplate(filename)
    content = []

    content.append(Paragraph(f"Patient: {name}"))
    content.append(Paragraph(f"Result: {result}"))
    content.append(Paragraph(f"Confidence: {confidence}%"))
    content.append(Paragraph(f"Stage: {stage}"))

    doc.build(content)

    return redirect('/' + filename)

# ================= RUN =================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)