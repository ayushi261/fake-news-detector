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

# ---------------------------------------------------------------------------
# OpenRouter configuration
# ---------------------------------------------------------------------------
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Any model tagged ":free" on https://openrouter.ai/models works here.
# Swap this constant if your chosen free model is retired/rate-limited.
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

# Optional but recommended by OpenRouter for free-tier usage/analytics.
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "http://localhost:5000")
OPENROUTER_SITE_NAME = os.environ.get("OPENROUTER_SITE_NAME", "VerifyX")


# Function to clean text (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def ai_fact_check(news_text):
    """
    Sends the article to OpenRouter and returns a concise AI fact-check
    explaining why the article looks real or fake, or stating that there
    is insufficient evidence to judge it.

    Returns a plain string (the explanation) in every case, including
    failure cases, so callers never have to special-case error handling.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("AI fact-check skipped: OPENROUTER_API_KEY is not set.")
        return "AI fact-check unavailable: server is missing OPENROUTER_API_KEY."

    prompt = f"""You are a fact-checking assistant. Evaluate the following news
article excerpt for factual accuracy and credibility signals (tone, sourcing,
sensationalism, verifiable claims, etc.).

Article excerpt:
\"\"\"{news_text[:2000]}\"\"\"

Respond in this exact format:
VERDICT: [Likely Real / Likely Fake / Insufficient Evidence]
EXPLANATION: [2-3 concise sentences explaining your reasoning. If there isn't
enough information to judge, say so explicitly instead of guessing.]"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_SITE_URL,
        "X-Title": OPENROUTER_SITE_NAME,
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 300,
    }

    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=20,
        )

        # Missing/invalid API key
        if response.status_code == 401:
            print("OpenRouter auth error: invalid or missing API key.")
            return "AI fact-check unavailable: invalid OpenRouter API key."

        # Rate limit exceeded
        if response.status_code == 429:
            print("OpenRouter rate limit hit.")
            return "AI fact-check unavailable: rate limit reached, please try again shortly."

        # Any other non-2xx status
        if not response.ok:
            print(f"OpenRouter error {response.status_code}: {response.text[:300]}")
            return "AI fact-check unavailable: the AI service returned an error."

        data = response.json()

        choices = data.get("choices")
        if not choices or "message" not in choices[0] or "content" not in choices[0]["message"]:
            print(f"OpenRouter unexpected response shape: {data}")
            return "AI fact-check unavailable: received an unexpected response format."

        explanation = choices[0]["message"]["content"].strip()
        return explanation if explanation else "AI fact-check unavailable: empty response from the AI service."

    except requests.exceptions.Timeout:
        print("OpenRouter request timed out.")
        return "AI fact-check unavailable: request to the AI service timed out."

    except requests.exceptions.RequestException as e:
        # Covers connection errors, DNS failures, etc.
        print(f"OpenRouter network error: {e}")
        return "AI fact-check unavailable: could not reach the AI service."

    except ValueError as e:
        # response.json() failed to parse
        print(f"OpenRouter returned invalid JSON: {e}")
        return "AI fact-check unavailable: received an invalid response from the AI service."

    except Exception as e:
        # Final safety net so /predict never crashes because of this call
        print(f"Unexpected error in ai_fact_check: {e}")
        return "AI fact-check unavailable due to an unexpected error."


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

        # AI fact-check via OpenRouter (replaces Gemini)
        fact_check_explanation = ai_fact_check(text)

        # Prepare response.
        # 'ai_fact_check' is the new field requested. 'gemini_check' is kept
        # as an alias with the same content so any existing frontend code
        # that still reads response.gemini_check keeps working unmodified.
        result = {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': predicted_label,
            'confidence': confidence_score,
            'real_probability': float(confidence[1] * 100),
            'fake_probability': float(confidence[0] * 100),
            'ai_fact_check': fact_check_explanation,
            'gemini_check': {'success': True, 'result': fact_check_explanation},
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