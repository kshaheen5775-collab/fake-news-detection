# Step 2: Text Preprocessing and Cleaning
# File: src/step2_preprocessing.py

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("FAKE NEWS DETECTION - TEXT PREPROCESSING")
print("="*60)

# ===== STEP 1: LOAD DATA =====
print("\n[1] Loading combined dataset...")

df = pd.read_csv('data/combined_news.csv')
print(f"   ✅ Loaded {len(df)} articles")

# ===== STEP 2: DOWNLOAD NLTK DATA =====
print("\n[2] Downloading NLTK data (if not already downloaded)...")

try:
    nltk.data.find('corpora/stopwords')
    print("   ✅ Stopwords already downloaded")
except:
    nltk.download('stopwords', quiet=True)
    print("   ✅ Stopwords downloaded")

try:
    nltk.data.find('tokenizers/punkt')
    print("   ✅ Punkt already downloaded")
except:
    nltk.download('punkt', quiet=True)
    print("   ✅ Punkt downloaded")

# ===== STEP 3: DEFINE PREPROCESSING FUNCTION =====
print("\n[3] Creating text preprocessing pipeline...")

def preprocess_text(text):
    """
    Clean and preprocess text:
    1. Lowercase
    2. Remove URLs
    3. Remove special characters
    4. Remove numbers
    5. Tokenization
    6. Remove stopwords
    7. Stemming
    """
    
    # Convert to string (in case of NaN)
    text = str(text)
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # 3. Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # 4. Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 5. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. Tokenization (split into words)
    words = text.split()
    
    # 7. Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # 8. Stemming (reduce words to root form)
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    # Join back into string
    cleaned_text = ' '.join(words)
    
    return cleaned_text

print("   ✅ Preprocessing function created")

# ===== STEP 4: SAMPLE PREPROCESSING DEMO =====
print("\n[4] Preprocessing Demo:")
print("-" * 60)

sample_text = df['text'].iloc[0][:200]  # First 200 chars
print(f"\n   📰 Original Text:")
print(f"   {sample_text}...")

cleaned_sample = preprocess_text(sample_text)
print(f"\n   ✨ Cleaned Text:")
print(f"   {cleaned_sample}...")

# ===== STEP 5: PREPROCESS ALL TEXTS =====
print("\n[5] Preprocessing all articles...")
print("   ⏳ This may take 2-3 minutes...")

# Combine title and text for better analysis
df['full_text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)

# Apply preprocessing
df['cleaned_text'] = df['full_text'].apply(preprocess_text)

print(f"   ✅ Preprocessing complete!")

# Check results
print(f"\n   📊 Cleaned text statistics:")
print(f"      Average words (original): {df['full_text'].str.split().str.len().mean():.0f}")
print(f"      Average words (cleaned): {df['cleaned_text'].str.split().str.len().mean():.0f}")
print(f"      Reduction: {(1 - df['cleaned_text'].str.split().str.len().mean() / df['full_text'].str.split().str.len().mean())*100:.1f}%")

# ===== STEP 6: REMOVE EMPTY TEXTS =====
print("\n[6] Removing empty texts after cleaning...")

before_count = len(df)
df = df[df['cleaned_text'].str.len() > 10]  # Remove very short texts
after_count = len(df)

print(f"   ✅ Removed {before_count - after_count} empty/too short articles")
print(f"   ✅ Remaining articles: {after_count}")

# ===== STEP 7: WORD CLOUDS =====
print("\n[7] Creating Word Clouds...")

# Separate fake and real news
fake_text = ' '.join(df[df['label'] == 1]['cleaned_text'])
real_text = ' '.join(df[df['label'] == 0]['cleaned_text'])

# Word Cloud for FAKE news
print("   ⏳ Generating Fake News word cloud...")
plt.figure(figsize=(12, 6))

wordcloud_fake = WordCloud(width=800, height=400, 
                           background_color='white',
                           colormap='Reds',
                           max_words=100).generate(fake_text)

plt.subplot(1, 2, 1)
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.title('FAKE NEWS - Most Common Words', fontsize=14, fontweight='bold', color='red')
plt.axis('off')

# Word Cloud for REAL news
print("   ⏳ Generating Real News word cloud...")
wordcloud_real = WordCloud(width=800, height=400,
                           background_color='white',
                           colormap='Greens',
                           max_words=100).generate(real_text)

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_real, interpolation='bilinear')
plt.title('REAL NEWS - Most Common Words', fontsize=14, fontweight='bold', color='green')
plt.axis('off')

plt.tight_layout()
plt.savefig('visualizations/04_wordclouds.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 04_wordclouds.png")
plt.close()

# ===== STEP 8: TOP WORDS ANALYSIS =====
print("\n[8] Analyzing most common words...")

from collections import Counter

def get_top_words(text, n=20):
    """Get top N most common words"""
    words = text.split()
    return Counter(words).most_common(n)

# Top words in fake news
top_fake = get_top_words(fake_text, 15)
print("\n   🔴 Top 15 words in FAKE news:")
for i, (word, count) in enumerate(top_fake, 1):
    print(f"      {i:2d}. {word:15s} ({count:5d} times)")

# Top words in real news
top_real = get_top_words(real_text, 15)
print("\n   🟢 Top 15 words in REAL news:")
for i, (word, count) in enumerate(top_real, 1):
    print(f"      {i:2d}. {word:15s} ({count:5d} times)")

# ===== STEP 9: VISUALIZE TOP WORDS =====
print("\n[9] Creating top words visualization...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Fake news top words
words_fake = [word for word, count in top_fake]
counts_fake = [count for word, count in top_fake]

ax1.barh(words_fake[::-1], counts_fake[::-1], color='#e74c3c')
ax1.set_xlabel('Frequency', fontsize=12)
ax1.set_title('Top 15 Words in FAKE News', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Real news top words
words_real = [word for word, count in top_real]
counts_real = [count for word, count in top_real]

ax2.barh(words_real[::-1], counts_real[::-1], color='#2ecc71')
ax2.set_xlabel('Frequency', fontsize=12)
ax2.set_title('Top 15 Words in REAL News', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/05_top_words.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 05_top_words.png")
plt.close()

# ===== STEP 10: SAVE PREPROCESSED DATA =====
print("\n[10] Saving preprocessed dataset...")

# Save only necessary columns
df_final = df[['title', 'cleaned_text', 'label']].copy()
df_final.to_csv('data/preprocessed_news.csv', index=False)

print(f"   ✅ Saved: data/preprocessed_news.csv")
print(f"   📊 Final dataset size: {len(df_final)} articles")

# ===== STEP 11: DATA STATISTICS =====
print("\n[11] Final Statistics:")
print("-" * 60)

label_counts = df_final['label'].value_counts()
print(f"\n   📊 Label Distribution:")
print(f"      Real News (0): {label_counts[0]:,} ({label_counts[0]/len(df_final)*100:.1f}%)")
print(f"      Fake News (1): {label_counts[1]:,} ({label_counts[1]/len(df_final)*100:.1f}%)")

print(f"\n   📝 Text Statistics:")
print(f"      Avg cleaned text length: {df_final['cleaned_text'].str.len().mean():.0f} chars")
print(f"      Avg words per article: {df_final['cleaned_text'].str.split().str.len().mean():.0f} words")

# ===== SUMMARY =====
print("\n" + "="*60)
print("✨ TEXT PREPROCESSING COMPLETE!")
print("="*60)
print(f"\n🎯 What We Did:")
print(f"   ✅ Cleaned and preprocessed {len(df_final):,} articles")
print(f"   ✅ Removed stopwords, special chars, numbers")
print(f"   ✅ Applied stemming")
print(f"   ✅ Created word clouds")
print(f"   ✅ Analyzed most common words")
print(f"   ✅ Saved preprocessed data")
print(f"\n📊 New Visualizations Created: 2")
print(f"   • 04_wordclouds.png")
print(f"   • 05_top_words.png")
print(f"\n🚀 Next Step: Feature Engineering (step3_feature_engineering.py)")
print("="*60)