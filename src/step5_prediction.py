# Step 5: Prediction System - Test Your Model!
# File: src/step5_prediction.py

import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np

print("="*60)
print("FAKE NEWS DETECTION - PREDICTION SYSTEM")
print("="*60)

# ===== STEP 1: LOAD SAVED MODELS =====
print("\n[1] Loading trained model and vectorizer...")

# Load TF-IDF vectorizer
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
print("   ✅ TF-IDF vectorizer loaded")

# Load best model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("   ✅ Best model loaded")

# Load model info
with open('models/model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)
print(f"   ✅ Model info loaded")
print(f"      Model: {model_info['model_name']}")
print(f"      Accuracy: {model_info['accuracy']*100:.2f}%")

# ===== STEP 2: DEFINE PREPROCESSING FUNCTION =====
print("\n[2] Setting up text preprocessing...")

def preprocess_text(text):
    """
    Preprocess text just like we did during training
    """
    # Convert to string
    text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

print("   ✅ Preprocessing function ready")

# ===== STEP 3: DEFINE PREDICTION FUNCTION =====
print("\n[3] Creating prediction function...")

def predict_news(news_text, show_details=True):
    """
    Predict if news is FAKE or REAL
    
    Args:
        news_text: The news article text
        show_details: Whether to print detailed output
    
    Returns:
        prediction, confidence
    """
    # Preprocess
    cleaned_text = preprocess_text(news_text)
    
    # Vectorize
    vectorized = tfidf.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(vectorized)[0]
    
    # Get confidence (probability)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(vectorized)[0]
        confidence = probabilities[prediction] * 100
    else:
        # For models without predict_proba (like SVM without probability=True)
        confidence = 0  # We'll use decision function or just mark as N/A
    
    if show_details:
        print("\n" + "="*60)
        if prediction == 1:
            print("🚨 PREDICTION: FAKE NEWS!")
            print("="*60)
        else:
            print("✅ PREDICTION: REAL NEWS!")
            print("="*60)
        
        print(f"\n   Label: {prediction} (0=Real, 1=Fake)")
        if confidence > 0:
            print(f"   Confidence: {confidence:.2f}%")
        
        print(f"\n   Original text (first 200 chars):")
        print(f"   {news_text[:200]}...")
        
        print(f"\n   Cleaned text (first 150 chars):")
        print(f"   {cleaned_text[:150]}...")
    
    return prediction, confidence

print("   ✅ Prediction function ready")

# ===== STEP 4: TEST WITH SAMPLE NEWS =====
print("\n" + "="*60)
print("[4] TESTING WITH SAMPLE NEWS ARTICLES")
print("="*60)

# Sample fake news articles
fake_samples = [
    "Breaking: Scientists discover aliens living on Mars! Government has been hiding this for years.",
    "SHOCKING: Celebrity caught in major scandal! You won't believe what happened next!",
    "Miracle cure found! This simple trick will make you lose 50 pounds in one week!"
]

# Sample real news articles
real_samples = [
    "The stock market closed higher today as investors responded positively to economic data.",
    "The government announced new education policies aimed at improving literacy rates nationwide.",
    "Scientists published a peer-reviewed study on climate change in the journal Nature today."
]

print("\n🔴 Testing with FAKE-LIKE news samples:")
print("-" * 60)
for i, sample in enumerate(fake_samples, 1):
    print(f"\n--- Sample {i} ---")
    pred, conf = predict_news(sample)

print("\n\n🟢 Testing with REAL-LIKE news samples:")
print("-" * 60)
for i, sample in enumerate(real_samples, 1):
    print(f"\n--- Sample {i} ---")
    pred, conf = predict_news(sample)

# ===== STEP 5: LOAD AND TEST REAL DATA =====
print("\n" + "="*60)
print("[5] TESTING WITH ACTUAL DATASET SAMPLES")
print("="*60)

# Load original data
df = pd.read_csv('data/combined_news.csv')

# Get random samples
np.random.seed(42)
sample_indices = np.random.choice(len(df), 5, replace=False)

print("\n📰 Testing with 5 random articles from dataset:\n")

correct_predictions = 0
for i, idx in enumerate(sample_indices, 1):
    article = df.iloc[idx]
    true_label = article['label']
    text = str(article['title']) + ' ' + str(article['text'])
    
    print(f"\n{'='*60}")
    print(f"Article {i}")
    print(f"{'='*60}")
    print(f"True Label: {'FAKE' if true_label == 1 else 'REAL'}")
    
    pred, conf = predict_news(text, show_details=False)
    
    print(f"Predicted: {'FAKE' if pred == 1 else 'REAL'}")
    if conf > 0:
        print(f"Confidence: {conf:.2f}%")
    
    is_correct = (pred == true_label)
    correct_predictions += is_correct
    
    status = "✅ CORRECT" if is_correct else "❌ WRONG"
    print(f"Status: {status}")
    
    print(f"\nArticle excerpt:")
    print(f"{text[:200]}...")

print(f"\n{'='*60}")
print(f"Accuracy on these samples: {correct_predictions}/5 ({correct_predictions*20}%)")
print(f"{'='*60}")

# ===== STEP 6: INTERACTIVE MODE =====
print("\n" + "="*60)
print("[6] INTERACTIVE PREDICTION MODE")
print("="*60)

def interactive_mode():
    """
    Interactive mode where user can input news and get predictions
    """
    print("\n🎯 You can now test the model with your own news!")
    print("   (Type 'quit' or 'exit' to stop)\n")
    
    while True:
        print("-" * 60)
        news_input = input("\n📰 Enter news text (or 'quit' to exit):\n> ")
        
        if news_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye! Thanks for using the Fake News Detector!")
            break
        
        if len(news_input.strip()) < 10:
            print("⚠️  Please enter a longer text (at least 10 characters)")
            continue
        
        predict_news(news_input)

print("\n💡 Interactive mode available!")
print("   To use it, uncomment the line below in the code:")
print("   # interactive_mode()")

# Uncomment the line below to enable interactive mode
# interactive_mode()

# ===== STEP 7: CREATE SUMMARY REPORT =====
print("\n" + "="*60)
print("[7] CREATING PROJECT SUMMARY REPORT")
print("="*60)

# Load test results
y_test = np.load('models/y_test.npy')
y_pred_file = 'models/best_model_predictions.npy'

# If predictions are saved, load them; otherwise use current model
try:
    y_pred = np.load(y_pred_file)
except:
    # Load test data and make predictions
    X_test = np.load('models/X_test.npy')
    y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

summary = {
    'Project': 'Fake News Detection System',
    'Model Used': model_info['model_name'],
    'Dataset Size': len(df),
    'Training Samples': len(df) - len(y_test),
    'Testing Samples': len(y_test),
    'Accuracy': accuracy_score(y_test, y_pred) * 100,
    'Precision': precision_score(y_test, y_pred) * 100,
    'Recall': recall_score(y_test, y_pred) * 100,
    'F1-Score': f1_score(y_test, y_pred) * 100
}

print("\n📊 PROJECT SUMMARY REPORT:")
print("-" * 60)
for key, value in summary.items():
    if isinstance(value, float):
        print(f"{key:20s}: {value:.2f}%")
    else:
        print(f"{key:20s}: {value}")

# Save summary
summary_df = pd.DataFrame([summary])
summary_df.to_csv('models/project_summary.csv', index=False)
print("\n   ✅ Saved: models/project_summary.csv")

# ===== FINAL SUMMARY =====
print("\n" + "="*60)
print("✨ PREDICTION SYSTEM COMPLETE!")
print("="*60)

print("\n🎯 Your Fake News Detector is Ready!")
print("\n✅ What You Can Do Now:")
print("   1. Test with any news article")
print("   2. Get instant FAKE/REAL predictions")
print("   3. See confidence scores")
print("   4. Use for presentations/demos")

print("\n📊 Model Performance:")
print(f"   • Accuracy: {summary['Accuracy']:.2f}%")
print(f"   • Precision: {summary['Precision']:.2f}%")
print(f"   • Recall: {summary['Recall']:.2f}%")
print(f"   • F1-Score: {summary['F1-Score']:.2f}%")

print("\n💾 All Files Saved:")
print("   • Models: 6 files in models/ folder")
print("   • Visualizations: 7 images in visualizations/ folder")
print("   • Data: 3 CSV files in data/ folder")

print("\n🚀 Next Steps:")
print("   1. Document your project (README.md)")
print("   2. Upload to GitHub")
print("   3. Add to your portfolio")
print("   4. Practice presentation")

print("\n🎉 CONGRATULATIONS! You built your first ML project!")
print("="*60)