from flask import Flask, jsonify, request
import cv2
import numpy as np
import joblib  # Changed from pickle to joblib
from flask_cors import CORS
import base64
import traceback  # For debugging errors

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Load the pre-trained model
try:
    model = joblib.load('./eye_state_model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", str(e))
    exit(1)  # Stop execution if model is not loaded

def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))  # Resize image to 64x64
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    frame = frame / 255.0  # Normalize pixel values (0 to 1)
    frame = frame.flatten()  # Convert (64,64) to (4096,)
    frame = frame.reshape(1, -1)  # Make it (1, 4096) for SVC
    return frame

@app.route('/detect-drowsiness', methods=['POST'])
def detect_drowsiness():
    try:
        # Get the image data from the frontend (base64 encoded)
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])  # Remove metadata prefix
        nparray = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparray, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Predict using the loaded model
        prediction = model.predict(processed_frame)

        # Interpret the model output
        if isinstance(prediction[0], (list, np.ndarray)):
            drowsy = prediction[0][0] > 0.5  # Assuming sigmoid activation for binary classification
        else:
            drowsy = int(prediction[0]) == 1  # If directly outputting class labels

        print(f'Prediction: {prediction}, Drowsy: {drowsy}')

        # Return the appropriate message
        message = "Eyes are closed" if drowsy else "Eyes are open"
        return jsonify({'drowsy': drowsy, 'message': message})
    
    except Exception as e:
        print("Error:", str(e))
        print(traceback.format_exc())  # Print full error traceback
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    
