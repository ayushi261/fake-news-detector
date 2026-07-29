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


# ---------------------------------------------------------------------------
# Wikipedia-based context lookup — free, no API key, no signup required
# ---------------------------------------------------------------------------
def get_wikipedia_context(text):
    """
    Searches Wikipedia for an article related to the claim/article text.
    ...
    """
    try:
        query = ' '.join(text.split()[:15])  # first ~15 words as the search query
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'format': 'json',
                'srlimit': 1
            },
            headers={
                'User-Agent': 'VerifyX-FakeNewsDetector/1.0 (https://verifyx-izeg.onrender.com; contact: ayushi1901singh@gmail.com)'
            },
            timeout=10
        )
        data = response.json()
        results = data.get('query', {}).get('search', [])
        if results:
            snippet = re.sub('<[^<]+?>', '', results[0]['snippet'])  # strip HTML tags
            return {'found': True, 'title': results[0]['title'], 'snippet': snippet}
        return {'found': False}
    except requests.exceptions.Timeout:
        print("Wikipedia lookup timed out.")
        return {'found': False, 'error': 'timeout'}
    except requests.exceptions.RequestException as e:
        print(f"Wikipedia network error: {e}")
        return {'found': False, 'error': 'network_error'}
    except Exception as e:
        print(f"Wikipedia lookup error: {e}")
        return {'found': False, 'error': str(e)}


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

        # Clean the text
        cleaned_text = clean_text(text)

        # Vectorize
        text_vectorized = vectorizer.transform([cleaned_text])

        # Predict (unchanged ML logic)
        prediction = model.predict(text_vectorized)[0]
        confidence = model.predict_proba(text_vectorized)[0]
        predicted_label = 'REAL' if prediction == 1 else 'FAKE'
        confidence_score = float(max(confidence))

        # Wikipedia context lookup — always runs, free, no key required
        wiki_context = get_wikipedia_context(text)

        if wiki_context.get('found'):
            note = f"Related Wikipedia article found: \"{wiki_context['title']}\". Compare this against the claim yourself."
        else:
            note = "No matching Wikipedia article found. This may be very recent news or an obscure claim — treat the prediction with extra caution."

        # Prepare response
        result = {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': predicted_label,
            'confidence': confidence_score,
            'real_probability': float(confidence[1] * 100),
            'fake_probability': float(confidence[0] * 100),
            'note': note,
            'wikipedia_context': wiki_context,
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
    print("\n" + "=" * 60)
    print("FAKE NEWS DETECTOR API - STARTING")
    print("=" * 60)
    print(f"API running on port: {port}")
    print("Endpoint: POST /predict")
    print("=" * 60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=port)