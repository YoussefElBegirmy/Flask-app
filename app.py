from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load your TensorFlow model
model = tf.keras.models.load_model('path_to_your_model.h5')

# Define the class labels
class_labels = [
    "Apple", "Avocado", "Banana", "Beetroot", "Blackberry", "Blueberry", "Broccoli", "Cabbage", "Capsicum", "Carrot",
    "Cauliflower", "Chilli Pepper", "Corn", "Cucumber", "Dates", "Dragonfruit", "Eggplant", "Fig", "Garlic", "Ginger",
    "Grapes", "Guava", "Jalapeno", "Kiwi", "Lemon", "Lettuce", "Mango", "Mushroom", "Okra", "Olive", "Onion", "Orange",
    "Paprika", "Peanuts", "Pear", "Peas", "Pineapple", "Pomegranate", "Potato", "Pumpkin", "Radish", "Rambutan", 
    "Soy Beans", "Spinach", "Strawberry", "Sweetcorn", "Sweet Potato", "Tomato", "Turnip", "Watermelon"
]

# Define the preprocessing and prediction function
def preprocess_and_predict(image_path, model, target_size=(300, 300)):
    """
    Preprocesses an image and predicts its class using the provided model.
    
    Args:
        image_path (str): Path to the image file.
        model (tf.keras.Model): Trained model to use for prediction.
        target_size (tuple): Target size for resizing the image (default is (300, 300)).
    
    Returns:
        str: Predicted class label.
    """
    # Load and preprocess the image
    image = tf.keras.utils.load_img(image_path, target_size=target_size)
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    preprocessed_image = tf.keras.applications.efficientnet.preprocess_input(image_array)
    
    # Predict the class probabilities
    predictions = model.predict(preprocessed_image)
    
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)
    
    # Get the predicted class label
    predicted_class_label = class_labels[predicted_class_index]
    
    return predicted_class_label

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save the uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        file.save(temp_path)
        
        # Preprocess the image and get predictions
        predicted_class_label = preprocess_and_predict(temp_path, model)
        
        # Return the prediction result
        return jsonify({'output': predicted_class_label})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
