from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained model and vectorizer
print("Loading model and vectorizer...")
with open('fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("✓ Model loaded successfully!")

# Function to clean text (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Home route
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Fake News Detector API',
        'version': '1.0',
        'endpoint': '/predict',
        'method': 'POST',
        'example': {
            'text': 'Your news article text here'
        }
    })

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get text from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Please provide text in JSON format'}), 400
        
        text = data['text']
        
        if len(text.strip()) == 0:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Vectorize
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(text_vectorized)[0]
        confidence = model.predict_proba(text_vectorized)[0]
        
        # Prepare response
        result = {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': 'REAL' if prediction == 1 else 'FAKE',
            'confidence': float(max(confidence) * 100),
            'real_probability': float(confidence[1] * 100),
            'fake_probability': float(confidence[0] * 100)
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check route
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'API is running'}), 200

# Run the app
if __name__ == '__main__':
    print("\n" + "="*60)
    print("FAKE NEWS DETECTOR API - STARTING")
    print("="*60)
    print("API running on: http://localhost:5000")
    print("Endpoint: POST http://localhost:5000/predict")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
    