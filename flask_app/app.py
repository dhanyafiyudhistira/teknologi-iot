from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import uuid
import unicodedata
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Inisialisasi Flask
app = Flask(__name__)
app.secret_key = "road_damage_classifier_secret_key"  # Diperlukan untuk flash messages
app.config['JSON_AS_ASCII'] = False  # Ensure Flask doesn't force ASCII encoding
app.config['DEFAULT_CHARSET'] = 'utf-8'  # Set default charset to UTF-8

# Load model klasifikasi
MODEL_PATH = 'saved_models/classifier_model.h5'
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {str(e).encode('utf-8', errors='ignore').decode('ascii', errors='ignore')}")
    model = None

# Bikin folder upload kalau belum ada
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Konfigurasi upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to normalize filenames and avoid encoding issues
def normalize_filename(filename):
    filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    return secure_filename(filename)

# Preprocessing image
def prepare_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(128, 128))  # Sesuaikan ukuran input model
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize
        return img_array
    except Exception as e:
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    error = None
    
    if request.method == 'POST':
        # Check if model is loaded
        if model is None:
            error = "Model not loaded. Please contact administrator."
            return render_template('index.html', error=error)

        # Check if file part exists
        if 'file' not in request.files:
            error = "No file part"
            return render_template('index.html', error=error)
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            error = "No file selected"
            return render_template('index.html', error=error)
        
        # Check file size
        if request.content_length > MAX_CONTENT_LENGTH:
            error = "File too large. Maximum size is 5MB."
            return render_template('index.html', error=error)

        # Check if file is allowed
        if file and allowed_file(file.filename):
            # Secure and normalize filename to prevent encoding issues
            original_filename = normalize_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            try:
                file.save(filepath)
                filename = unique_filename
                
                # Process image and make prediction
                img = prepare_image(filepath)
                if img is not None:
                    pred = model.predict(img)
                    pred_label = "Damaged" if pred[0][0] >= 0.5 else "Not Damaged"
                    prediction = pred_label
                else:
                    error = "Error processing image"
                    os.remove(filepath)  # Clean up file if processing fails
            except Exception as e:
                # Sanitize error message to avoid encoding issues
                error = f"An error occurred: {str(e).encode('utf-8', errors='ignore').decode('ascii', errors='ignore')}"
                if os.path.exists(filepath):
                    os.remove(filepath)  # Clean up file if saving fails
        else:
            error = "File type not allowed. Please upload JPG, JPEG, or PNG images only."
    
    return render_template('index.html', prediction=prediction, filename=filename, error=error)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)