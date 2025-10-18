#!/usr/bin/env python3
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# load data
df = pd.read_csv('data/train.csv')
target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# train
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X, y)

# save
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
print("Logistic Regression model saved to models/model.pkl")
