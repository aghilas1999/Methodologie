from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import requests
from pre_process1 import preprocess_image
from ImageDataPreparation import load_data
import os

app = Flask(__name__)

model = tf.keras.models.load_model("../models/Aghilas_modele.h5")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        print("file : ", file)
        file_path = "temp_image.jpeg"
        file.save(file_path)

        img = load_data(file_path)
        print("img : ", img)

        processed_image = preprocess_image(img)

        print("processed image : ", processed_image)
        prediction = model.predict(processed_image)

        if prediction > 0.5:
            result = "Pneumonia"
        else:
            result = "Normal"

        os.remove(file_path)

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)