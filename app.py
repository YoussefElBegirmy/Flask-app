from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import requests
from io import BytesIO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define class labels
CLASS_LABELS = ["Apple", "Avocado", "Banana", "Beetroot", "Blackberry", "Blueberry", "Broccoli", "Cabbage", 
                "Capsicum", "Carrot", "Cauliflower", "Chilli Pepper", "Corn", "Cucumber", "Dates", "Dragonfruit", 
                "Eggplant", "Fig", "Garlic", "Ginger", "Grapes", "Guava", "Jalapeno", "Kiwi", "Lemon", "Lettuce", 
                "Mango", "Mushroom", "Okra", "Olive", "Onion", "Orange", "Paprika", "Peanuts", "Pear", "Peas", 
                "Pineapple", "Pomegranate", "Potato", "Pumpkin", "Radish", "Rambutan", "Soy Beans", "Spinach", 
                "Strawberry", "Sweetcorn", "Sweet Potato", "Tomato", "Turnip", "Watermelon"]

# Load TFLite model from Hugging Face
def load_model_from_huggingface(model_url):
    response = requests.get(model_url)
    response.raise_for_status()  # Ensure the request was successful
    model_data = BytesIO(response.content)

    # Save the model locally for the TFLite interpreter
    with open("model.tflite", "wb") as f:
        f.write(model_data.read())

    # Load the model with the TFLite interpreter
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

# URL of your model on Hugging Face
HUGGINGFACE_MODEL_URL = "https://huggingface.co/your-model-url/model.tflite"
interpreter = load_model_from_huggingface(HUGGINGFACE_MODEL_URL)

# Get input/output details for the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define prediction function
def predict(image):
    # Preprocess the image
    image = image.resize((300, 300))  # Resize to model input size
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array /= 255.0  # Normalize the image

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)

    # Run inference
    interpreter.invoke()

    # Get the prediction
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(predictions)
    predicted_label = CLASS_LABELS[predicted_index]

    return predicted_label

# Define Flask endpoint
@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    try:
        # Open the image file
        image = Image.open(file.stream)

        # Predict the class label
        predicted_label = predict(image)
        return jsonify({"output": predicted_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
