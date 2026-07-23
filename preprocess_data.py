import pandas as pd
import numpy as np
import re
import os

# Step 1: Load the datasets
print("Loading datasets...")
true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

# Step 2: Add labels (1 = Real, 0 = Fake)
true_df['label'] = 1
fake_df['label'] = 0

# Step 3: Combine both datasets
df = pd.concat([true_df, fake_df], ignore_index=True)
print(f"Total articles: {len(df)}")

# Step 4: Combine title and text (full article)
df['content'] = df['title'] + " " + df['text']

# Step 5: Clean text function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Step 6: Apply cleaning
print("Cleaning text...")
df['content'] = df['content'].apply(clean_text)

# Step 7: Remove empty rows
df = df[df['content'].str.len() > 0]

# Step 8: Select only needed columns
df = df[['content', 'label']]

# Step 9: Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 10: Save to new CSV
df.to_csv('cleaned_news_data.csv', index=False)
print(f"Cleaned data saved! Total rows: {len(df)}")
print("First few rows:")
print(df.head())