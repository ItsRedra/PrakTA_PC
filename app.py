from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
import joblib
from werkzeug.utils import secure_filename

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_glcm_features(image_path):
    image = imread(image_path)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = rgb2gray(image)
    image = (image * 255).astype(np.uint8)
    # Use single distance and angle to reduce feature dimensionality
    distances = [1]
    angles = [0]
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = {}
    for prop in props:
        feat = graycoprops(glcm, prop)[0, 0]
        features[prop] = feat
    return features

def load_model():
    knn = joblib.load("model/knn_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    le = joblib.load("model/label_encoder.pkl")
    return knn, le, scaler

knn_model, label_encoder, scaler = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        features = extract_glcm_features(filepath)
        features_scaled = scaler.transform(np.array(list(features.values())).reshape(1, -1))
        pred_encoded = knn_model.predict(features_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        prob = knn_model.predict_proba(features_scaled)[0]
        confidence = max(prob) * 100
        return render_template('result.html', filename=filename, prediction=pred_label, confidence=confidence, feature_values=features)
    return redirect(request.url)

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/asset/<path:filename>')
def asset(filename):
    return send_from_directory('asset', filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
