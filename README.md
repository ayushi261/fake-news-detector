# TruthAI: Fake News Detector - AI-Powered Misinformation Detection

An intelligent machine learning system that detects fake news with 99% accuracy using natural language processing and classification algorithms.

## Features

- 99% Accuracy - Trained on 45,000+ real and fake news articles
- Real-time Predictions - Instant analysis with confidence scores
- Beautiful UI - Dark-themed, attention-grabbing interface
- REST API - Easy integration with web/mobile apps
- Cross-platform - Works on Windows, Mac, Linux

## Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 99.01% |
| Precision | 98.67% |
| Recall | 99.25% |
| F1-Score | 98.96% |

## Architecture

Data Collection (45K+ articles)
         |
Data Preprocessing (Clean & Label)
         |
Machine Learning Model (Logistic Regression)
         |
Flask REST API
         |
Beautiful Frontend (HTML/CSS/JS)

## Tech Stack

Backend:
- Python 3.13
- Flask (REST API)
- scikit-learn (Machine Learning)
- pandas, numpy (Data Processing)

Frontend:
- HTML5
- CSS3
- JavaScript (Vanilla)

Data:
- 45,889 labeled news articles
- TF-IDF Vectorization
- Binary Classification

## Installation & Setup

Prerequisites:
- Python 3.8+
- pip (Python package manager)

Local Setup:

1. Clone the repository
git clone https://github.com/ayushi261/fake-news-detector.git
cd fake-news-detector

2. Create virtual environment
python -m venv venv
venv\Scripts\activate  (Windows)
source venv/bin/activate  (Mac/Linux)

3. Install dependencies
pip install -r requirements.txt

4. Run the application
python app.py

5. Open in browser
Navigate to file:///path/to/index.html
Or open index.html directly

## Usage

1. Paste news article in the text area
2. Click "ANALYZE NOW"
3. Get instant prediction:
   - REAL NEWS or FAKE NEWS
   - Confidence percentage
   - Real vs Fake probability breakdown

Example Inputs:

Real News:
Scientists at Stanford University discover breakthrough in renewable energy technology with 47% efficiency improvement...

Fake News:
SHOCKING! 5G towers control people's minds! Microchips in vaccines confirmed by Area 51!

## Dataset

- Source: Kaggle (fake-and-real-news-dataset)
- Size: 45,889 labeled articles
- Split: 80% training, 20% testing
- Classes: Real News (21,417) | Fake News (23,472)

## Model Training Process

1. Data Loading - Import 45K+ articles
2. Text Cleaning - Remove URLs, special characters, normalize text
3. Vectorization - Convert text to numerical features using TF-IDF
4. Model Training - Train Logistic Regression classifier
5. Evaluation - Test on unseen data (99% accuracy!)
6. Serialization - Save model and vectorizer for deployment

## Project Structure

fake-news-detector/
├── app.py                      (Flask API)
├── train_model.py             (Model training script)
├── preprocess_data.py         (Data preprocessing)
├── index.html                 (Frontend interface)
├── fake_news_model.pkl        (Trained model)
├── tfidf_vectorizer.pkl       (Text vectorizer)
├── cleaned_news_data.csv      (Processed dataset)
├── requirements.txt           (Python dependencies)
├── Procfile                   (Deployment config)
└── README.md                  (This file)

## What I Learned

- Machine Learning: Data preprocessing, feature extraction, model training
- NLP: Text cleaning, TF-IDF vectorization, classification
- Backend: REST API design, Flask framework
- Frontend: Interactive UI with real-time feedback
- DevOps: Git, GitHub, deployment pipeline
- Data Science: 45K+ dataset handling, model evaluation

## Future Enhancements

- Mobile app (React Native/Flutter)
- Real-time news feed analysis
- User feedback loop for model improvement
- Multi-language support
- Advanced deep learning models (BERT, GPT)
- Cloud deployment with auto-scaling

## API Documentation

Endpoint: /predict

Method: POST

Request:
{
  "text": "Your news article text here"
}

Response:
{
  "prediction": "REAL",
  "confidence": 95.32,
  "real_probability": 95.32,
  "fake_probability": 4.68,
  "text": "Your news article text here..."
}

## Contributing

Feel free to fork this project and submit pull requests!

## License

This project is open source and available under the MIT License.

## Author

Ayushi Singh
- GitHub: @ayushi261
- Email: ayushi1901singh@gmail.com

Made with love for detecting misinformation