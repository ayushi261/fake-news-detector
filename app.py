from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import re
import os
import requests

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained model and vectorizer
print("Loading model and vectorizer...")
with open('fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("Model loaded successfully!")

# Function to clean text (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Gemini-based fact-check function
def check_with_gemini(text, api_key):
    """Ask Gemini to fact-check the claim"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

        prompt = f"""You are a fact-checking assistant. Evaluate this claim or article excerpt for factual accuracy based on your knowledge:

"{text[:500]}"

Respond in this exact format:
VERDICT: [TRUE / FALSE / UNVERIFIABLE / PARTIALLY TRUE]
EXPLANATION: [2-3 sentence explanation of your reasoning]"""

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        response = requests.post(url, json=payload, timeout=15)
        data = response.json()

        if 'candidates' in data and len(data['candidates']) > 0:
            output_text = data['candidates'][0]['content']['parts'][0]['text']
            return {'success': True, 'result': output_text.strip()}
        else:
            return {'success': False, 'error': data.get('error', {}).get('message', 'No response from Gemini')}

    except Exception as e:
        print(f"Gemini error: {e}")
        return {'success': False, 'error': str(e)}

# Home route - serves the frontend
@app.route('/', methods=['GET'])
def home():
    return send_from_directory('templates', 'index.html')

# API info route
@app.route('/api', methods=['GET'])
def api_info():
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

        # Optional Gemini fact-check
        api_key = data.get('googleApiKey', '')
        gemini_check = None

        if api_key:
            gemini_check = check_with_gemini(text, api_key)

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
            'fake_probability': float(confidence[0] * 100),
            'gemini_check': gemini_check
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
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*60)
    print("FAKE NEWS DETECTOR API - STARTING")
    print("="*60)
    print(f"API running on port: {port}")
    print("Endpoint: POST /predict")
    print("="*60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=port)