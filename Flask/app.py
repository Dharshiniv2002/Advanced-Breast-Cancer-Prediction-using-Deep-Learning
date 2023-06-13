import os
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf

app = Flask(__name__)
model_path = "Flask/breastcancer.h5"
model = None

# Load the model
def load_model():
    global model
    model = tf.keras.models.load_model(model_path, compile=False)

# Preprocess the image
def preprocess_image(image):
    img = image.resize((64, 64))
    x = np.array(img)
    x = x / 255.0  # Normalize the image
    x = np.expand_dims(x, axis=0)
    return x

@app.route('/', methods=['GET'])
def bcancer():
    return render_template('bcancer.html')

@app.route('/predict.html', methods=['GET'])
def predict():
    return render_template('predict.html')

@app.route('/pred', methods=['POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image uploaded"

        file = request.files['image']
        if file.filename == '':
            return "No image selected"

        if file:
            try:
                image = Image.open(file)
                x = preprocess_image(image)

                if model is None:
                    load_model()
                pred = model.predict(x)
                classes = np.argmax(pred, axis=1)

                if classes[0] == 0:
                    text = "The tumor is benign. No need to worry!"
                else:
                    text = "It is a malignant tumor. Please consult a doctor."

                return text

            except Exception as e:
                return str(e)

if __name__ == '__main__':
    app.run(debug=True)