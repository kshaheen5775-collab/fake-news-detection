# Step 4: Machine Learning Model Training
# File: src/step4_model_training.py

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
import time
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("FAKE NEWS DETECTION - MODEL TRAINING")
print("="*60)

# ===== STEP 1: LOAD DATA =====
print("\n[1] Loading processed data...")

X_train = np.load('models/X_train.npy')
X_test = np.load('models/X_test.npy')
y_train = np.load('models/y_train.npy')
y_test = np.load('models/y_test.npy')

print(f"   ✅ Training data: {X_train.shape}")
print(f"   ✅ Testing data: {X_test.shape}")
print(f"   ✅ Training labels: {len(y_train)}")
print(f"   ✅ Testing labels: {len(y_test)}")

# ===== STEP 2: DEFINE MODELS =====
print("\n[2] Initializing machine learning models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='linear', random_state=42)
}

print(f"   ✅ Initialized {len(models)} models:")
for i, name in enumerate(models.keys(), 1):
    print(f"      {i}. {name}")

# ===== STEP 3: TRAIN AND EVALUATE MODELS =====
print("\n[3] Training and evaluating models...")
print("   ⏳ This may take 3-5 minutes (especially SVM)...")
print("-" * 60)

results = {}

for name, model in models.items():
    print(f"\n   🤖 Training: {name}")
    
    # Start timer
    start_time = time.time()
    
    # Train model
    print(f"      ⏳ Fitting model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    print(f"      ⏳ Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # End timer
    training_time = time.time() - start_time
    
    # Store results
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'training_time': training_time
    }
    
    # Print results
    print(f"      ✅ Training complete! (Time: {training_time:.2f}s)")
    print(f"      📊 Accuracy: {accuracy*100:.2f}%")
    print(f"      📊 Precision: {precision*100:.2f}%")
    print(f"      📊 Recall: {recall*100:.2f}%")
    print(f"      📊 F1-Score: {f1*100:.2f}%")

# ===== STEP 4: COMPARISON TABLE =====
print("\n" + "="*60)
print("[4] MODEL COMPARISON RESULTS")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy']*100 for m in results.keys()],
    'Precision': [results[m]['precision']*100 for m in results.keys()],
    'Recall': [results[m]['recall']*100 for m in results.keys()],
    'F1-Score': [results[m]['f1_score']*100 for m in results.keys()],
    'Time (s)': [results[m]['training_time'] for m in results.keys()]
})

print("\n" + comparison_df.to_string(index=False))

# Find best model
best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Accuracy: {results[best_model_name]['accuracy']*100:.2f}%")

# ===== STEP 5: DETAILED CLASSIFICATION REPORT =====
print("\n" + "="*60)
print(f"[5] DETAILED REPORT - {best_model_name}")
print("="*60)

y_pred_best = results[best_model_name]['predictions']
print("\n" + classification_report(y_test, y_pred_best, 
                                   target_names=['Real News', 'Fake News']))

# ===== STEP 6: CONFUSION MATRICES =====
print("\n[6] Creating confusion matrices...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')

for idx, (name, result) in enumerate(results.items()):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    cm = confusion_matrix(y_test, result['predictions'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    
    ax.set_title(f"{name}\nAccuracy: {result['accuracy']*100:.2f}%", 
                fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('visualizations/06_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 06_confusion_matrices.png")
plt.close()

# ===== STEP 7: PERFORMANCE COMPARISON CHART =====
print("\n[7] Creating performance comparison chart...")

fig, ax = plt.subplots(figsize=(12, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.2

for idx, (name, result) in enumerate(results.items()):
    values = [
        result['accuracy']*100,
        result['precision']*100,
        result['recall']*100,
        result['f1_score']*100
    ]
    ax.bar(x + idx*width, values, width, label=name)

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics)
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 105])

plt.tight_layout()
plt.savefig('visualizations/07_model_comparison.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 07_model_comparison.png")
plt.close()

# ===== STEP 8: SAVE BEST MODEL =====
print("\n[8] Saving best model...")

best_model = results[best_model_name]['model']

with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"   ✅ Saved: models/best_model.pkl")
print(f"   📌 Best model: {best_model_name}")

# Save model info
model_info = {
    'model_name': best_model_name,
    'accuracy': results[best_model_name]['accuracy'],
    'precision': results[best_model_name]['precision'],
    'recall': results[best_model_name]['recall'],
    'f1_score': results[best_model_name]['f1_score'],
    'training_time': results[best_model_name]['training_time']
}

with open('models/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print(f"   ✅ Saved: models/model_info.pkl")

# ===== STEP 9: EXAMPLE PREDICTIONS =====
print("\n[9] Testing model with sample predictions...")
print("-" * 60)

# Get 5 random samples
sample_indices = np.random.choice(len(X_test), 5, replace=False)

print("\n   📰 Sample Predictions:\n")
for i, idx in enumerate(sample_indices, 1):
    true_label = "FAKE" if y_test[idx] == 1 else "REAL"
    pred_label = "FAKE" if y_pred_best[idx] == 1 else "REAL"
    
    status = "✅" if y_test[idx] == y_pred_best[idx] else "❌"
    
    print(f"   {i}. True: {true_label:4s} | Predicted: {pred_label:4s} {status}")

# Calculate correct predictions
correct = (y_test == y_pred_best).sum()
total = len(y_test)
print(f"\n   📊 Total Correct: {correct}/{total} ({correct/total*100:.2f}%)")

# ===== STEP 10: ERROR ANALYSIS =====
print("\n[10] Error Analysis...")

# False Positives (Real news predicted as Fake)
false_positives = ((y_test == 0) & (y_pred_best == 1)).sum()
# False Negatives (Fake news predicted as Real)
false_negatives = ((y_test == 1) & (y_pred_best == 0)).sum()

print(f"   ❌ False Positives: {false_positives} (Real news incorrectly marked as Fake)")
print(f"   ❌ False Negatives: {false_negatives} (Fake news incorrectly marked as Real)")
print(f"   ✅ True Positives: {((y_test == 1) & (y_pred_best == 1)).sum()}")
print(f"   ✅ True Negatives: {((y_test == 0) & (y_pred_best == 0)).sum()}")

# ===== SUMMARY =====
print("\n" + "="*60)
print("✨ MODEL TRAINING COMPLETE!")
print("="*60)

print(f"\n🎯 Training Summary:")
print(f"   ✅ Trained {len(models)} different models")
print(f"   ✅ Best model: {best_model_name}")
print(f"   ✅ Best accuracy: {results[best_model_name]['accuracy']*100:.2f}%")
print(f"   ✅ Tested on {len(y_test):,} articles")

print(f"\n📊 Best Model Performance:")
print(f"   • Accuracy: {results[best_model_name]['accuracy']*100:.2f}%")
print(f"   • Precision: {results[best_model_name]['precision']*100:.2f}%")
print(f"   • Recall: {results[best_model_name]['recall']*100:.2f}%")
print(f"   • F1-Score: {results[best_model_name]['f1_score']*100:.2f}%")

print(f"\n💾 Files Saved:")
print(f"   • models/best_model.pkl")
print(f"   • models/model_info.pkl")
print(f"   • visualizations/06_confusion_matrices.png")
print(f"   • visualizations/07_model_comparison.png")

print(f"\n🚀 Next Step: Create Prediction Function (step5_prediction.py)")
print("="*60)