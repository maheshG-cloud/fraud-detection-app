import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

# Create simple training data
data = pd.DataFrame({
    'amount': [100, 200, 3000, 4000, 150, 5000],
    'is_online': [0, 1, 1, 1, 0, 1],
    'is_foreign': [0, 0, 1, 1, 0, 1],
    'is_high_risk_device': [0, 0, 1, 1, 0, 1],
    'fraud': [0, 0, 1, 1, 0, 1]
})

X = data[['amount', 'is_online', 'is_foreign', 'is_high_risk_device']]
y = data['fraud']

model = LogisticRegression()
model.fit(X, y)

# Save model
with open('model/fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully!")