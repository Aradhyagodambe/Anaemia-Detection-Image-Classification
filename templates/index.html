import os
import cv2
import numpy as np
import joblib
import albumentations as A
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload size

# Ensure the uploads folder exists within the static directory
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Data Augmentation function
def augment_image(img):
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5)
    ])
    return transform(image=img)['image']

# Feature Extraction function
def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Unable to read image. Check the file format.")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (50, 50))
    img = augment_image(img)
    
    # 3D color histogram (8x8x8 = 512 features)
    features = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
    return features.flatten().reshape(1, -1)

# Load the trained SVM model
MODEL_PATH = "anemia_svm_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', error="Model file 'anemia_svm_model.pkl' not found.")
        
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Extract features and predict
            features = extract_features(file_path)
            prediction_val = model.predict(features)[0]
            
            result_text = "Anemic" if prediction_val == 1 else "Non-Anemic"
            
            # Use forward slashes for web URL paths
            img_url = f"/static/uploads/{filename}"

            return render_template('index.html', prediction=result_text, img_url=img_url)
        except Exception as e:
            return render_template('index.html', error=str(e))

if __name__ == "__main__":
    # Render assigns the PORT environment variable dynamically
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
