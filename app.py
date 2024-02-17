import os
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.preprocessing import image
from training import Meso4 
import base64
from io import BytesIO


app = Flask(__name__)

# Load the pre-trained model
meso = Meso4()
meso.load('Meso4_DF')

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get file from the request
    file = request.files['file']

    # Save the file to a temporary location
    file_path = 'temp.jpg'  # Choose a temporary file path
    file.save(file_path)

    # Load the saved image
    img = image.load_img(file_path, target_size=(256, 256))
    os.remove(file_path)  # Remove the temporary file

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = img_array / 255.0

    # Make prediction
    prediction = meso.predict(preprocessed_img)[0][0]

    # Interpret prediction
    if prediction > 0.5:
        result = "REAL"
    else:
        result = "FAKE"

    # Render the same template with the result and image
    img_name = file.filename
    img_id = hash(file.read())
    img_data = encode_image(img)
    return render_template('index.html', result=result, img_data=img_data, img_name=img_name, img_id=img_id)

def encode_image(img):
    # Convert the image to base64 encoding
    from io import BytesIO
    from PIL import Image
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

if __name__ == '__main__':
    app.run(debug=True)
