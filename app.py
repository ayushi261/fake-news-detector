from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import re
import os
import requests

app = Flask(__name__)
CORS(app)

print("Loading model and vectorizer...")
with open('fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("Model loaded successfully!")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def check_with_gemini(text, api_key):
    """Ask Gemini to fact-check the claim. Fails gracefully on quota/auth issues."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

        prompt = f"""You are a fact-checking assistant. Evaluate this claim or article excerpt for factual accuracy based on your knowledge:

"{text[:500]}"

Respond in this exact format:
VERDICT: [TRUE / FALSE / UNVERIFIABLE / PARTIALLY TRUE]
EXPLANATION: [2-3 sentence explanation of your reasoning]"""

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(url, json=payload, timeout=15)
        data = response.json()

        if 'candidates' in data and len(data['candidates']) > 0:
            output_text = data['candidates'][0]['content']['parts'][0]['text']
            return {'success': True, 'result': output_text.strip()}

        raw_error = data.get('error', {}).get('message', 'No response from Gemini')

        if 'quota' in raw_error.lower():
            friendly = ("This Gemini API key has no free-tier quota allocated yet. "
                        "This is a Google account-side limit, not an app error — "
                        "try a different key or check billing status at "
                        "aistudio.google.com/apikey.")
        elif 'not found' in raw_error.lower() or 'api key not valid' in raw_error.lower():
            friendly = "This API key appears to be invalid. Please check it at aistudio.google.com/apikey."
        else:
            friendly = raw_error

        return {'success': False, 'error': friendly}

    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Gemini took too long to respond. Please try again.'}
    except Exception as e:
        print(f"Gemini error: {e}")
        return {'success': False, 'error': 'Could not reach the fact-checking service.'}


@app.route('/', methods=['GET'])
def home():
    return send_from_directory('templates', 'index.html')


@app.route('/api', methods=['GET'])
def api_info():
    return jsonify({
        'message': 'Fake News Detector API',
        'version': '1.0',
        'endpoint': '/predict',
        'method': 'POST',
        'example': {'text': 'Your news article text here'}
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Please provide text in JSON format'}), 400

        text = data['text']

        if len(text.strip()) == 0:
            return jsonify({'error': 'Text cannot be empty'}), 400

        api_key = data.get('googleApiKey', '')
        gemini_check = None
        if api_key:
            gemini_check = check_with_gemini(text, api_key)

        cleaned_text = clean_text(text)
        text_vectorized = vectorizer.transform([cleaned_text])

        prediction = model.predict(text_vectorized)[0]
        confidence = model.predict_proba(text_vectorized)[0]

        # Flag low-confidence / very short inputs, since the model is
        # least reliable on these (style-based, not fact-based classifier)
        is_short_input = len(text.strip()) < 60
        max_conf = float(max(confidence) * 100)

        result = {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': 'REAL' if prediction == 1 else 'FAKE',
            'confidence': max_conf,
            'real_probability': float(confidence[1] * 100),
            'fake_probability': float(confidence[0] * 100),
            'low_reliability_warning': is_short_input or max_conf < 70,
            'gemini_check': gemini_check
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'API is running'}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)