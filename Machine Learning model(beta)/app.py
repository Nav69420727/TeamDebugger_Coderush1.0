from flask import Flask, request, jsonify
import pickle
import numpy as np
import mne

app = Flask(__name__)

# Manual CORS handling
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Define the required function
def eeg_power_band(epochs):
    """
    EEG feature extraction function.
    """
    FREQ_BANDS = {
        "delta": [0.5, 4.5],
        "theta": [4.5, 8.5],
        "alpha": [8.5, 11.5],
        "sigma": [11.5, 15.5],
        "beta": [15.5, 30]
    }

    spectrum = epochs.compute_psd(picks='eeg', fmin=0.5, fmax=30.0, n_fft=1000)
    psds, freqs = spectrum.get_data(return_freqs=True)
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)

# Custom unpickler
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'eeg_power_band':
            return eeg_power_band
        return super().find_class(module, name)

# Load the model
try:
    with open('model.pkl', 'rb') as file:
        model = CustomUnpickler(file).load()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return "Flask EEG Model Server is Running!"

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'eeg_data' not in data:
            return jsonify({'error': 'No EEG data provided'}), 400
        
        # Placeholder for actual preprocessing
        processed_data = preprocess_eeg_data(data['eeg_data'])
        
        # Mock prediction (replace with actual model prediction)
        prediction = [0]  # Example prediction
        confidence = 0.85  # Example confidence
        
        return jsonify({
            'prediction': int(prediction[0]),
            'confidence': float(confidence),
            'message': 'Schizophrenia detected' if prediction[0] == 1 else 'Normal EEG'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def preprocess_eeg_data(raw_data):
    """
    Preprocess EEG data - implement your actual logic here
    """
    # Placeholder implementation
    return raw_data

if __name__ == '__main__':
    app.run(debug=True)