import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB

print("=" * 60)
print("FAKE NEWS DETECTOR - MODEL TRAINING")
print("=" * 60)

# Step 1: Load cleaned data
print("\n1. Loading cleaned data...")
df = pd.read_csv('cleaned_news_data.csv')
print(f"   Total samples: {len(df)}")
print(f"   Real news: {(df['label'] == 1).sum()}")
print(f"   Fake news: {(df['label'] == 0).sum()}")

# Step 2: Split into training (80%) and testing (20%)
print("\n2. Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    df['content'], 
    df['label'], 
    test_size=0.2, 
    random_state=42
)
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# Step 3: Vectorize text (convert words to numbers)
print("\n3. Converting text to numbers (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"   Features created: {X_train_vec.shape[1]}")

# Step 4: Train Logistic Regression model
print("\n4. Training Logistic Regression model...")
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_vec, y_train)
print("   Model training complete!")

# Step 5: Make predictions
print("\n5. Making predictions on test data...")
y_pred = model.predict(X_test_vec)

# Step 6: Evaluate model
print("\n6. Evaluating model performance...")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"   Accuracy:  {accuracy * 100:.2f}%")
print(f"   Precision: {precision * 100:.2f}%")
print(f"   Recall:    {recall * 100:.2f}%")
print(f"   F1-Score:  {f1 * 100:.2f}%")

# Step 7: Save the model
print("\n7. Saving trained model...")
with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   Model saved as 'fake_news_model.pkl'")

# Step 8: Save the vectorizer
print("\n8. Saving vectorizer...")
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("   Vectorizer saved as 'tfidf_vectorizer.pkl'")

# Step 9: Test with sample predictions
print("\n9. Testing with sample predictions...")
sample_texts = [
    "breaking news president announces new policy reform today",
    "fake aliens found in area 51 government conspiracy confirmed"
]

for i, text in enumerate(sample_texts):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    confidence = model.predict_proba(text_vec)[0]
    
    label = "REAL" if prediction == 1 else "FAKE"
    print(f"\n   Sample {i+1}: '{text[:50]}...'")
    print(f"   Prediction: {label}")
    print(f"   Confidence: {max(confidence) * 100:.2f}%")

print("\n" + "=" * 60)
print("TRAINING COMPLETE! Model is ready for deployment.")
print("=" * 60)