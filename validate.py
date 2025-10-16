#!/usr/bin/env python3
import json
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import os

# load model and test data
if not os.path.exists('models/model.pkl'):
    raise FileNotFoundError("Model not found at models/model.pkl — run train stage first.")

test_df = pd.read_csv('data/test_encoded.csv')
# detect target column (last column)
target_col = test_df.columns[-1]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

model = joblib.load('models/model.pkl')
print("Loaded model")

# predict & metrics
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
report = classification_report(y_test, preds, output_dict=True)

# save metrics (test only)
with open('metrics.json', 'w') as f:
    json.dump({'test_accuracy': float(acc), 'classification_report': report}, f)
print(f"Saved metrics.json — test_accuracy: {acc:.4f}")

# confusion matrix plot
labels = np.unique(y_test)
cm = confusion_matrix(y_test, preds, labels=labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Heart Disease')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")
