from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import re
import os
import requests

DEFAULT_SEARCH_ENGINE_ID = "8505974bd0e0d4ac0"
def search_google(query, api_key, search_engine_id=None):
    search_engine_id = search_engine_id or DEFAULT_SEARCH_ENGINE_ID
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': api_key,
            'cx': search_engine_id,
            'q': query,
            'num': 3
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'items' in data:
            results = []
            for item in data['items'][:3]:
                results.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })
            return results
        return []
    except Exception as e:
        print(f"Search error: {e}")
        return []
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
@app.route("/")
def home():
    return render_template("index.html")
    

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

        # Optional Google Search verification
        google_results = []
        api_key = data.get('googleApiKey', '')
        
        if api_key:
            search_query = text[:100]
            google_results = search_google(search_query, api_key)
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
            'google_results': google_results
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
    app.run(
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 5000))
)