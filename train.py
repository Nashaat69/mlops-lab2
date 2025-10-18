#!/usr/bin/env python3
import os
import joblib
import pandas as pd
from sklearn.svm import SVC

# load data
df = pd.read_csv('data/train_encoded.csv')
target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# train
# note: SVC benefits from scaled numeric features; preprocess.py already scales numeric cols.
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model.fit(X, y)

# save
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
print("✅ SVC model saved to models/model.pkl")
