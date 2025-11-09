# Step 3: Feature Engineering using TF-IDF
# File: src/step3_feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os

print("="*60)
print("FAKE NEWS DETECTION - FEATURE ENGINEERING")
print("="*60)

# Create models folder if doesn't exist
os.makedirs('models', exist_ok=True)

# ===== STEP 1: LOAD PREPROCESSED DATA =====
print("\n[1] Loading preprocessed data...")

df = pd.read_csv('data/preprocessed_news.csv')
print(f"   ✅ Loaded {len(df)} preprocessed articles")

# Check for any missing values
if df['cleaned_text'].isnull().sum() > 0:
    print(f"   ⚠️  Found {df['cleaned_text'].isnull().sum()} null values, removing...")
    df = df.dropna(subset=['cleaned_text'])
    print(f"   ✅ Cleaned dataset: {len(df)} articles")

# ===== STEP 2: PREPARE DATA =====
print("\n[2] Preparing features and labels...")

X = df['cleaned_text']  # Features (text)
y = df['label']         # Labels (0=Real, 1=Fake)

print(f"   ✅ Features (X): {len(X)} text samples")
print(f"   ✅ Labels (y): {len(y)} labels")
print(f"      - Real News (0): {(y==0).sum()}")
print(f"      - Fake News (1): {(y==1).sum()}")

# ===== STEP 3: TF-IDF VECTORIZATION =====
print("\n[3] Converting text to TF-IDF features...")
print("   ℹ️  What is TF-IDF?")
print("      TF-IDF converts text into numerical vectors")
print("      that machines can understand and learn from.")

print("\n   ⏳ Creating TF-IDF vectorizer...")

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,      # Use top 5000 most important words
    min_df=2,               # Word must appear in at least 2 documents
    max_df=0.8,             # Ignore words that appear in >80% documents
    ngram_range=(1, 2),     # Use single words and word pairs
    stop_words='english'    # Additional stopword filtering
)

print("   ✅ Vectorizer created")
print(f"      - Max features: 5000")
print(f"      - N-grams: (1, 2) - words and word pairs")

print("\n   ⏳ Fitting and transforming text data...")
print("      This may take 1-2 minutes...")

X_tfidf = tfidf.fit_transform(X)

print(f"   ✅ TF-IDF transformation complete!")
print(f"      - Matrix shape: {X_tfidf.shape}")
print(f"      - Features: {X_tfidf.shape[1]} TF-IDF values per article")
print(f"      - Sparsity: {(1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])) * 100:.2f}%")

# ===== STEP 4: SHOW TOP FEATURES =====
print("\n[4] Most important features (words):")

# Get feature names
feature_names = tfidf.get_feature_names_out()

print(f"   📝 Total vocabulary: {len(feature_names)} words/phrases")
print(f"\n   🔤 Sample features (first 20):")
for i, feature in enumerate(feature_names[:20], 1):
    print(f"      {i:2d}. {feature}")

# ===== STEP 5: TRAIN-TEST SPLIT =====
print("\n[5] Splitting data into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # For reproducibility
    stratify=y          # Maintain label distribution
)

print(f"   ✅ Data split complete!")
print(f"\n   📊 Training Set:")
print(f"      - Samples: {X_train.shape[0]}")
print(f"      - Real: {(y_train==0).sum()}")
print(f"      - Fake: {(y_train==1).sum()}")

print(f"\n   📊 Testing Set:")
print(f"      - Samples: {X_test.shape[0]}")
print(f"      - Real: {(y_test==0).sum()}")
print(f"      - Fake: {(y_test==1).sum()}")

print(f"\n   📈 Split Ratio:")
print(f"      - Training: {X_train.shape[0]/len(df)*100:.1f}%")
print(f"      - Testing: {X_test.shape[0]/len(df)*100:.1f}%")

# ===== STEP 6: SAVE PROCESSED DATA =====
print("\n[6] Saving processed data and vectorizer...")

# Save the vectorizer (we'll need it for new predictions)
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("   ✅ Saved: models/tfidf_vectorizer.pkl")

# Save train-test split data
np.save('models/X_train.npy', X_train.toarray())
np.save('models/X_test.npy', X_test.toarray())
np.save('models/y_train.npy', y_train)
np.save('models/y_test.npy', y_test)

print("   ✅ Saved: models/X_train.npy")
print("   ✅ Saved: models/X_test.npy")
print("   ✅ Saved: models/y_train.npy")
print("   ✅ Saved: models/y_test.npy")

# ===== STEP 7: DATA STATISTICS =====
print("\n[7] Final Statistics:")
print("-" * 60)

print(f"\n   📊 Dataset Overview:")
print(f"      Total articles: {len(df):,}")
print(f"      Training samples: {X_train.shape[0]:,} ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"      Testing samples: {X_test.shape[0]:,} ({X_test.shape[0]/len(df)*100:.1f}%)")

print(f"\n   🎯 Feature Information:")
print(f"      TF-IDF features: {X_tfidf.shape[1]:,}")
print(f"      Vocabulary size: {len(feature_names):,} words/phrases")

print(f"\n   ⚖️  Label Balance:")
train_balance = (y_train==1).sum() / len(y_train) * 100
test_balance = (y_test==1).sum() / len(y_test) * 100
print(f"      Training - Fake: {train_balance:.1f}%, Real: {100-train_balance:.1f}%")
print(f"      Testing  - Fake: {test_balance:.1f}%, Real: {100-test_balance:.1f}%")

# ===== STEP 8: EXAMPLE TRANSFORMATION =====
print("\n[8] Example: How text becomes numbers...")
print("-" * 60)

sample_text = X.iloc[0]
print(f"\n   📰 Original cleaned text (first 100 chars):")
print(f"   '{sample_text[:100]}...'")

sample_vector = X_tfidf[0]
print(f"\n   🔢 TF-IDF vector representation:")
print(f"   Shape: {sample_vector.shape}")
print(f"   Non-zero values: {sample_vector.nnz}")

# Show top 5 TF-IDF values for this document
sample_dense = sample_vector.toarray()[0]
top_indices = sample_dense.argsort()[-5:][::-1]
print(f"\n   🔝 Top 5 important words in this article:")
for i, idx in enumerate(top_indices, 1):
    if sample_dense[idx] > 0:
        print(f"      {i}. '{feature_names[idx]}' (score: {sample_dense[idx]:.4f})")

# ===== SUMMARY =====
print("\n" + "="*60)
print("✨ FEATURE ENGINEERING COMPLETE!")
print("="*60)

print(f"\n🎯 What We Did:")
print(f"   ✅ Converted {len(df):,} articles to TF-IDF vectors")
print(f"   ✅ Created {X_tfidf.shape[1]:,} numerical features")
print(f"   ✅ Split data: {X_train.shape[0]:,} train, {X_test.shape[0]:,} test")
print(f"   ✅ Saved vectorizer and processed data")

print(f"\n💾 Files Saved:")
print(f"   • models/tfidf_vectorizer.pkl")
print(f"   • models/X_train.npy")
print(f"   • models/X_test.npy")
print(f"   • models/y_train.npy")
print(f"   • models/y_test.npy")

print(f"\n🚀 Next Step: Model Training (step4_model_training.py)")
print(f"   Now we're ready to train machine learning models!")
print("="*60)