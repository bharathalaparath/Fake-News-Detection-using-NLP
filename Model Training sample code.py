# Required Libraries
# pip install pandas numpy scikit-learn nltk matplotlib seaborn joblib

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Purpose of the Downloads
# stopwords: Downloads a list of common words (e.g., 'the', 'a', 'is') in various languages that are often removed during text preprocessing.
# wordnet: Downloads the WordNet lexical database, which is used for tasks like lemmatization and finding synonyms and hypernyms.
# omw-1.4 (Open Multilingual Wordnet): Downloads multilingual WordNet data, which is required by the wordnet package for full functionality and to support lemmatization across different languages. 

print("\nFAKE NEWS DETECTION MODEL - TRAINING PIPELINE")

print("\n# ============================================")
print("# 1. LOAD AND EXPLORE DATA")
print("# ============================================\n")

# df = pd.read_csv('C:\My Projects\Fake News Detection using NLP\Test Data\english_fake_news_2212.csv')
df = pd.read_csv('C:\My Projects\Fake News Detection using NLP\Test Data\Dataset01.csv')

print(f"\nInitital Dataset shape: {df.shape}")
print(f"Initial Dataset - Column names: {df.columns.tolist()}")
print(f"Class distribution:\n{df['label'].value_counts()}")
print(f"Sample records from the dataset:\n {df.head(3)}")

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Handle missing values
df['headline'] = df['headline'].fillna('')
df['body_text'] = df['body_text'].fillna('')


print("\n# ============================================")
print("# 2. TEXT PREPROCESSING")
print("# ============================================\n")

def preprocess_text(text):
    # """Clean and preprocess text data"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # print("# ")
    text = re.sub(r'\s+', ' ', text).strip()
    # print("# ")
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # print("# ")
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

print("\n# Combine headline and body_text")
df['text'] = df['headline'] + ' ' + df['body_text']
print(f"\n News Articles :\n{df['text'].iloc[0]}")

print("\n# Apply preprocessing to the combined text data")
print("# Convert to lowercase\t# Remove URLs\t# Remove special characters and digits\n# Remove extra whitespace\t# Remove stopwords\t# Lemmatization")
df['cleaned_text'] = df['text'].apply(preprocess_text)

print("\nText preprocessing complete!")
print(f"Sample cleaned text: {df['cleaned_text'].iloc[0][:200]}")
# .iloc[0] - Selects the first row (integer position 0) of that column.
# [:200] - Slices the string to show only the first 200 characters. 

print(f"\nCleaned Articles Data set:\n{df['cleaned_text']}")
print(f"Class distribution:\n{df['label'].value_counts()}")

print("\n============================================")
print("3. TRAIN-TEST SPLIT")
print("============================================")

# Article Text and Feature Labels
X = df['cleaned_text']
y = df['label']

print("\nStratified Train-Test Split (80-20)")
# split a dataset (X for features, y for target labels) into training and testing subsets. 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# test_size=0.2: Reserves 20% of the dataset for testing and uses the remaining 80% for training.
# random_state=42: Sets a seed for the random number generator, ensuring that the split is exactly the same every time you run the code.
# stratify=y: Ensures that the proportion of classes in both the training and testing sets matches the proportion in the original dataset y. This is crucial for classification tasks with imbalanced data to ensure the model trains on a representative sample. 

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Train class distribution:\n{y_train.value_counts()}")

print("\n============================================")
print("4. FEATURE EXTRACTION (TF-IDF)")
print("============================================")

# TF-IDF Vectorization with improved parameters
tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,      # Limit vocabulary size to 5000 words
    min_df=2,               # Ignore words appearing in fewer than 2 documents
    max_df=0.8,             # Ignore words appearing in more than 80% of documents
    ngram_range=(1, 2)      # Include unigrams and bigrams
)

# Fit on training data and transform / Learns the vocabulary and computes TF‑IDF scores.
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
# Transform test data using the same vocabulary / 
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"\nTF-IDF feature matrix shape (Train): {X_train_tfidf.shape}")
print(f"TF-IDF feature matrix shape (test): {X_test_tfidf.shape}")

print("\n============================================")
print("5. TRAIN MULTIPLE MODELS")
print("============================================")

# Define models to train
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine':  sklearn.svm.SVC(probability=True, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)

    print("# Predictions")  
    y_pred = model.predict(X_test_tfidf)
    # Generates the final class labels (e.g., 0 or 1, "spam" or "not spam") for the input data.
    # It selects the class with the highest probability by default.
    
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1] if hasattr(model, 'predict_proba') else None
    # model.predict_proba(): Returns the raw probability estimates for each possible class (e.g., [0.2, 0.8] for 20% chance of class 0, 80% chance of class 1).
    # [:, 1]: Extracts only the probabilities for the positive class (class 1).
    # hasattr(model, 'predict_proba'): Checks if the model supports probability estimates (e.g., Logistic Regression does, but LinearSVC does not by default).
    # else None: If the model doesn't support it, it assigns None to avoid an error. 

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"{name} Accuracy: {accuracy:.4f}")


print("\n============================================")
print("6. DETAILED EVALUATION")
print("============================================\n")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f"\nBest Model: {best_model_name}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"Prediction: {best_predictions}")

# Classification Report
print(f"\nClassification Report for {best_model_name}:\n")
print(classification_report(y_test, best_predictions, target_names=['Real News', 'Fake News']))

# Confusion Matrix
cm = confusion_matrix(y_test, best_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real News', 'Fake News'],
            yticklabels=['Real News', 'Fake News'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.savefig('confusion_matrix_' + best_model_name + '.png', dpi=300, bbox_inches='tight')
print(f"\nConfusion matrix saved as 'confusion_matrix_{best_model_name}.png'")