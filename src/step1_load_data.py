# Step 1: Load and Explore Fake News Dataset
# File: src/step1_load_data.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*60)
print("FAKE NEWS DETECTION PROJECT - DATA LOADING")
print("="*60)

# Create visualizations folder if doesn't exist
os.makedirs('visualizations', exist_ok=True)

# ===== STEP 1: LOAD DATA =====
print("\n[1] Loading datasets...")

try:
    # Load fake news
    fake = pd.read_csv('data/Fake.csv')
    print(f"   ✅ Fake news loaded: {len(fake)} articles")
    
    # Load real news
    true = pd.read_csv('data/True.csv')
    print(f"   ✅ Real news loaded: {len(true)} articles")
    
except FileNotFoundError as e:
    print(f"   ❌ Error: Dataset files not found!")
    print(f"   Make sure Fake.csv and True.csv are in 'data/' folder")
    exit()

# ===== STEP 2: ADD LABELS =====
print("\n[2] Adding labels...")

fake['label'] = 1  # 1 for FAKE
true['label'] = 0  # 0 for REAL

print("   ✅ Labels added (0=Real, 1=Fake)")

# ===== STEP 3: COMBINE DATASETS =====
print("\n[3] Combining datasets...")

df = pd.concat([fake, true], axis=0, ignore_index=True)
print(f"   ✅ Total articles: {len(df)}")

# ===== STEP 4: BASIC INFORMATION =====
print("\n[4] Dataset Information:")
print("-" * 60)

print(f"\n   📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n   📋 Columns: {list(df.columns)}")

print("\n   🏷️  Label Distribution:")
label_counts = df['label'].value_counts()
print(f"      Real News (0): {label_counts[0]} ({label_counts[0]/len(df)*100:.1f}%)")
print(f"      Fake News (1): {label_counts[1]} ({label_counts[1]/len(df)*100:.1f}%)")

print("\n   ❓ Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("      ✅ No missing values!")
else:
    print(missing[missing > 0])

# ===== STEP 5: FIRST FEW ROWS =====
print("\n[5] First 3 Articles:")
print("-" * 60)

for i in range(3):
    row = df.iloc[i]
    label = "FAKE" if row['label'] == 1 else "REAL"
    print(f"\n   Article {i+1} [{label}]:")
    print(f"   Title: {row['title'][:80]}...")
    if 'text' in df.columns:
        print(f"   Text: {str(row['text'])[:100]}...")

# ===== STEP 6: TEXT LENGTH ANALYSIS =====
print("\n[6] Analyzing text lengths...")

if 'text' in df.columns:
    df['text_length'] = df['text'].astype(str).str.len()
    df['word_count'] = df['text'].astype(str).str.split().str.len()
    
    print("\n   Average Text Length:")
    print(f"      Real News: {df[df['label']==0]['text_length'].mean():.0f} characters")
    print(f"      Fake News: {df[df['label']==1]['text_length'].mean():.0f} characters")
    
    print("\n   Average Word Count:")
    print(f"      Real News: {df[df['label']==0]['word_count'].mean():.0f} words")
    print(f"      Fake News: {df[df['label']==1]['word_count'].mean():.0f} words")

# ===== STEP 7: VISUALIZATIONS =====
print("\n[7] Creating visualizations...")

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Visualization 1: Label Distribution
plt.figure(figsize=(8, 5))
colors = ['#2ecc71', '#e74c3c']  # Green for Real, Red for Fake
label_counts.plot(kind='bar', color=colors)
plt.title('Fake vs Real News Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Label (0=Real, 1=Fake)', fontsize=12)
plt.ylabel('Number of Articles', fontsize=12)
plt.xticks([0, 1], ['Real News', 'Fake News'], rotation=0)
for i, v in enumerate(label_counts):
    plt.text(i, v + 100, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/01_label_distribution.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 01_label_distribution.png")
plt.close()

# Visualization 2: Text Length Comparison
if 'text_length' in df.columns:
    plt.figure(figsize=(10, 6))
    
    real_lengths = df[df['label']==0]['text_length']
    fake_lengths = df[df['label']==1]['text_length']
    
    plt.hist([real_lengths, fake_lengths], bins=50, 
             label=['Real News', 'Fake News'],
             color=['#2ecc71', '#e74c3c'], alpha=0.7)
    
    plt.title('Text Length Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Text Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/02_text_length_distribution.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: 02_text_length_distribution.png")
    plt.close()

# Visualization 3: Box Plot for Text Length
if 'text_length' in df.columns:
    plt.figure(figsize=(8, 6))
    
    sns.boxplot(data=df, x='label', y='text_length', palette=['#2ecc71', '#e74c3c'])
    
    plt.title('Text Length Comparison: Fake vs Real', fontsize=16, fontweight='bold')
    plt.xlabel('News Type', fontsize=12)
    plt.ylabel('Text Length (characters)', fontsize=12)
    plt.xticks([0, 1], ['Real News', 'Fake News'])
    plt.tight_layout()
    plt.savefig('visualizations/03_text_length_boxplot.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: 03_text_length_boxplot.png")
    plt.close()

# ===== STEP 8: SAVE COMBINED DATASET =====
print("\n[8] Saving combined dataset...")

df.to_csv('data/combined_news.csv', index=False)
print("   ✅ Saved: data/combined_news.csv")

# ===== SUMMARY =====
print("\n" + "="*60)
print("✨ DATA LOADING COMPLETE!")
print("="*60)
print(f"\n📊 Summary:")
print(f"   • Total Articles: {len(df)}")
print(f"   • Real News: {label_counts[0]}")
print(f"   • Fake News: {label_counts[1]}")
print(f"   • Visualizations Created: 3")
print(f"   • Combined Dataset Saved: ✅")
print("\n🎯 Next Step: Text Preprocessing (step2_preprocessing.py)")
print("="*60)