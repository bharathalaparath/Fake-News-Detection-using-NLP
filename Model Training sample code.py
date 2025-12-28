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

print("FAKE NEWS DETECTION MODEL - TRAINING PIPELINE")

########################################################
# 1. LOAD AND EXPLORE DATA
# ============================================

print("\n[1] Loading dataset...")
df = pd.read_csv('C:\My Projects\Fake News Detection using NLP\Test Data\english_fake_news_2212.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")
print(f"\nSample records:")
print(df.head(3))

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Handle missing values
df['headline'] = df['headline'].fillna('')
df['body_text'] = df['body_text'].fillna('')


#######################################################
# 2. TEXT PREPROCESSING
# ============================================

print("\n[2] Preprocessing text data...")

def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

# Combine headline and body_text
df['text'] = df['headline'] + ' ' + df['body_text']
print(df['text'])

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)

# print("Text preprocessing complete!")
print(f"Sample cleaned text:\n{df['cleaned_text'].iloc[0][:200]}...")

print(f"Cleaned Articles Data set:\n{df['cleaned_text']}")
print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {df['cleaned_text'].columns.tolist()}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")

# ============================================
# 3. TRAIN-TEST SPLIT
# ============================================

print("\n[3] Splitting data into train and test sets...")

X = df['cleaned_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Train class distribution:\n{y_train.value_counts()}")

# ============================================
# 4. FEATURE EXTRACTION (TF-IDF)
# ============================================

print("\n[4] Extracting features using TF-IDF...")

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,      # Limit vocabulary size
    min_df=2,               # Ignore words appearing in fewer than 2 documents
    max_df=0.8,             # Ignore words appearing in more than 80% of documents
    ngram_range=(1, 2)      # Include unigrams and bigrams
)

# Fit on training data and transform / Learns the vocabulary and computes TF‑IDF scores.
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
# Transform test data using the same vocabulary / 
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")
print(f"TF-IDF feature matrix shape (test): {X_test_tfidf.shape}")


# ============================================
# 5. TRAIN MULTIPLE MODELS
# ============================================

print("\n[5] Training multiple classification models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1] if hasattr(model, 'predict_proba') else None

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"{name} Accuracy: {accuracy:.4f}")


# ============================================
# 6. DETAILED EVALUATION
# ============================================

print("\n[6] Detailed Model Evaluation...")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f"\nBest Model: {best_model_name}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")

# Classification Report
print(f"\nClassification Report for {best_model_name}:")
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

plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved as 'confusion_matrix.png'")

